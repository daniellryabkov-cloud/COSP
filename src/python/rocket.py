"""
rocket.py - Multi-stage launch vehicle simulation.

Implements:
  • Pre-launch state
  • Vertical rise phase (clear launch tower)
  • Gravity turn (pitch-over manoeuvre)
  • Max-Q throttle reduction
  • Stage separation events
  • Fairing jettison
  • CubeSat deployment into target orbit
  • Full 2-D trajectory integration with RK4
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from physics import (
    R_EARTH, MU_EARTH, TARGET_RADIUS, TARGET_ALTITUDE,
    gravity_acceleration, drag_force, atmospheric_density,
    orbital_elements, orbital_velocity, runge_kutta_4,
)
from stage import Stage, make_stage_1, make_stage_2


# ── Launch vehicle geometry / payload ─────────────────────────────────────────

FAIRING_MASS       = 120.0    # kg
CUBESAT_MASS       = 8.0      # kg   (dry + full propellant)
PAYLOAD_ADAPTER_M  = 20.0     # kg

GRAVITY_TURN_START_ALT = 1_500.0    # m   start pitch-over after this altitude
GRAVITY_TURN_END_ALT   = 70_000.0   # m   complete pitch-over by this altitude
VERTICAL_RISE_ALT      = 500.0      # m   pure vertical rise before pitch
MAX_Q_THROTTLE         = 0.75       # throttle during max-dynamic-pressure
MAX_Q_ALT_LOW          = 10_000.0   # m   max-Q throttle zone
MAX_Q_ALT_HIGH         = 30_000.0   # m

# Second burn target: circularise at target altitude
COAST_ALTITUDE         = 480_000.0  # m   start circularisation burn
CIRC_BURN_DV           = 50.0       # m/s  fine-tuning circularisation


@dataclass
class MissionEvent:
    time:   float
    name:   str
    detail: str


class Rocket:
    """
    Two-stage launch vehicle carrying a 6U CubeSat.

    State vector: [x, y, vx, vy]
      x = horizontal (East in 2-D)
      y = vertical (altitude from Earth centre)

    Launch site: equatorial, East-facing (optimum for LEO).
    Initial position: [0, R_EARTH].
    """

    # ── Mission phases ────────────────────────────────────────────────────────
    PHASE_PRELAUNCH     = 'PRELAUNCH'
    PHASE_LIFTOFF       = 'LIFTOFF'
    PHASE_VERTICAL      = 'VERTICAL_RISE'
    PHASE_GRAVITY_TURN  = 'GRAVITY_TURN'
    PHASE_STAGE_SEP_1   = 'STAGE_1_SEPARATION'
    PHASE_STAGE_2_BURN  = 'STAGE_2_BURN'
    PHASE_COAST         = 'COAST'
    PHASE_CIRC_BURN     = 'CIRCULARISATION_BURN'
    PHASE_CUBESAT_SEP   = 'CUBESAT_SEPARATION'
    PHASE_COMPLETE      = 'MISSION_COMPLETE'

    def __init__(self):
        # ── Stages ────────────────────────────────────────────────────────────
        self.stage1 = make_stage_1()
        self.stage2 = make_stage_2()
        self._current_stage_idx = 0    # 0 = stage1, 1 = stage2, 2 = none

        # ── Structural masses ─────────────────────────────────────────────────
        self._fairing_mass   = FAIRING_MASS
        self._payload_mass   = CUBESAT_MASS + PAYLOAD_ADAPTER_M
        self._fairing_jettisoned = False
        self._payload_deployed   = False

        # ── State vector ──────────────────────────────────────────────────────
        # Initial: on launch pad at (0, R_EARTH), velocity = Earth surface rotation
        v_surface = 465.1      # m/s (equatorial surface speed)
        self._state = np.array([
            0.0,       R_EARTH,     # position [m]
            v_surface, 0.0,         # velocity [m/s]  (eastward surface rotation)
        ], dtype=float)

        # ── Flight parameters ─────────────────────────────────────────────────
        self._time        = 0.0
        self._phase       = self.PHASE_PRELAUNCH
        self._pitch_angle = np.pi / 2.0   # radians from +X; starts vertical (90°)
        self._throttle    = 0.0

        # ── Aerodynamics ─────────────────────────────────────────────────────
        self._cross_section = np.pi * (0.95)**2   # m²  stage-1 diameter

        # ── History ────────────────────────────────────────────────────────────
        self._history_time:     List[float]       = []
        self._history_pos:      List[np.ndarray]  = []
        self._history_vel:      List[np.ndarray]  = []
        self._history_alt:      List[float]       = []
        self._history_speed:    List[float]       = []
        self._history_accel:    List[float]       = []
        self._history_mass:     List[float]       = []
        self._history_phase:    List[str]         = []
        self._history_thrust:   List[float]       = []
        self._history_drag:     List[float]       = []
        self._events:           List[MissionEvent]= []

        # Deployment state (set externally after separation)
        self._cubesat_position: Optional[np.ndarray] = None
        self._cubesat_velocity: Optional[np.ndarray] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self._state[0:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._state[2:4].copy()

    @property
    def altitude(self) -> float:
        return np.linalg.norm(self._state[0:2]) - R_EARTH

    @property
    def speed(self) -> float:
        return np.linalg.norm(self._state[2:4])

    @property
    def time(self) -> float:
        return self._time

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def is_mission_complete(self) -> bool:
        return self._phase == self.PHASE_COMPLETE

    @property
    def cubesat_deployed(self) -> bool:
        return self._payload_deployed

    @property
    def cubesat_deployment_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._cubesat_position, self._cubesat_velocity

    def total_mass(self) -> float:
        m = 0.0
        if self._current_stage_idx == 0:
            m += self.stage1.total_mass + self.stage2.total_mass
        elif self._current_stage_idx == 1:
            m += self.stage2.total_mass
        # Payload
        if not self._payload_deployed:
            m += self._payload_mass
        if not self._fairing_jettisoned:
            m += self._fairing_mass
        return max(m, 1.0)

    # ── Guidance ──────────────────────────────────────────────────────────────

    def _gravity_turn_pitch(self) -> float:
        """
        Compute pitch angle (from +X toward +Y) during gravity turn.

        Gravity turn: pitch angle follows the velocity vector.
        We interpolate from 90° (vertical) to the optimal pitchover angle.

        Returns
        -------
        float  Pitch angle [rad].
        """
        alt = self.altitude

        if alt < VERTICAL_RISE_ALT:
            return np.pi / 2.0    # straight up

        if alt < GRAVITY_TURN_START_ALT:
            return np.pi / 2.0

        if alt > GRAVITY_TURN_END_ALT:
            # Follow velocity vector (natural gravity turn)
            vel = self.velocity
            if np.linalg.norm(vel) > 10.0:
                return np.arctan2(vel[1], vel[0])
            return np.pi / 4.0

        # Interpolate pitch-over from vertical to 15° flight path angle
        t = (alt - GRAVITY_TURN_START_ALT) / (GRAVITY_TURN_END_ALT - GRAVITY_TURN_START_ALT)
        t = np.clip(t, 0.0, 1.0)
        # Ease function for smoother pitch-over
        t_ease = t * t * (3.0 - 2.0 * t)

        angle_start = np.pi / 2.0         # 90°  vertical
        angle_end   = np.radians(15.0)    # 15°  above horizontal
        return angle_start + t_ease * (angle_end - angle_start)

    def _compute_throttle(self) -> float:
        """Max-Q throttle reduction."""
        alt = self.altitude
        if self.PHASE_VERTICAL in [self._phase, self.PHASE_LIFTOFF]:
            return 1.0
        if MAX_Q_ALT_LOW < alt < MAX_Q_ALT_HIGH:
            return MAX_Q_THROTTLE
        return 1.0

    # ── Dynamics ──────────────────────────────────────────────────────────────

    def _thrust_vector(self, thrust_magnitude: float) -> np.ndarray:
        """
        Convert scalar thrust and pitch angle to 2-D thrust vector.
        Thrust is aligned with pitch_angle in the local frame,
        which we project into ECI-like [x, y] components.
        """
        # Pitch angle is measured from horizontal (+X) toward vertical (+Y)
        return thrust_magnitude * np.array([
            np.cos(self._pitch_angle),
            np.sin(self._pitch_angle),
        ])

    def _derivatives(self, state: np.ndarray,
                     thrust: float,
                     cross_section: float) -> np.ndarray:
        pos = state[0:2]
        vel = state[2:4]
        alt = np.linalg.norm(pos) - R_EARTH
        m   = self.total_mass()

        # Gravity
        a_grav = gravity_acceleration(pos)

        # Thrust (along pitch direction in local frame)
        if thrust > 0 and m > 0:
            a_thrust = self._thrust_vector(thrust) / m
        else:
            a_thrust = np.zeros(2)

        # Drag
        a_drag = drag_force(vel, alt, cross_section, 0.35) / m if m > 0 else np.zeros(2)

        a_total = a_grav + a_thrust + a_drag
        return np.array([vel[0], vel[1], a_total[0], a_total[1]])

    # ── Stage management ──────────────────────────────────────────────────────

    def _active_stage(self) -> Optional[Stage]:
        if self._current_stage_idx == 0:
            return self.stage1
        elif self._current_stage_idx == 1:
            return self.stage2
        return None

    def _separate_stage1(self):
        self._log_event('STAGE_1_SEP', f'Stage-1 separated at alt={self.altitude/1000:.1f} km, '
                                        f'v={self.speed:.0f} m/s')
        self.stage1.shutdown()
        self._current_stage_idx = 1
        self._cross_section = np.pi * (0.85)**2
        self._phase = self.PHASE_STAGE_SEP_1
        # Brief event; actual phase transition happens next step

    def _jettison_fairing(self):
        if not self._fairing_jettisoned:
            self._fairing_jettisoned = True
            self._log_event('FAIRING_JETTISON',
                            f'Fairing jettisoned at alt={self.altitude/1000:.1f} km')

    def _deploy_cubesat(self):
        if not self._payload_deployed:
            self._payload_deployed   = True
            self._cubesat_position   = self.position.copy()
            self._cubesat_velocity   = self.velocity.copy()
            self._log_event('CUBESAT_DEPLOY',
                            f'CubeSat deployed at alt={self.altitude/1000:.1f} km, '
                            f'v={self.speed:.0f} m/s')
            self._phase = self.PHASE_CUBESAT_SEP

    # ── Event log ─────────────────────────────────────────────────────────────

    def _log_event(self, name: str, detail: str):
        evt = MissionEvent(self._time, name, detail)
        self._events.append(evt)
        print(f"  [T+{self._time:7.1f}s] EVENT: {name} — {detail}")

    # ── Phase state machine ────────────────────────────────────────────────────

    def _update_phase(self, thrust_produced: float):
        alt = self.altitude
        vel = self.velocity
        pos = self.position

        if self._phase == self.PHASE_PRELAUNCH:
            pass   # Handled by ignite()

        elif self._phase == self.PHASE_LIFTOFF:
            if alt > VERTICAL_RISE_ALT:
                self._phase = self.PHASE_VERTICAL

        elif self._phase == self.PHASE_VERTICAL:
            if alt > GRAVITY_TURN_START_ALT:
                self._phase = self.PHASE_GRAVITY_TURN
                self._log_event('GRAVITY_TURN_START',
                                f'Pitch-over commenced at alt={alt/1000:.1f} km')

        elif self._phase == self.PHASE_GRAVITY_TURN:
            # Jettison fairing above 90 km
            if alt > 90_000 and not self._fairing_jettisoned:
                self._jettison_fairing()

            # Stage 1 exhaustion → separate
            if (self._current_stage_idx == 0 and
                    (self.stage1._is_exhausted or self.stage1.propellant_fraction < 0.02)):
                self._separate_stage1()

        elif self._phase == self.PHASE_STAGE_SEP_1:
            # Ignite stage 2
            self.stage2.ignite()
            self._phase = self.PHASE_STAGE_2_BURN
            self._log_event('STAGE_2_IGNITION',
                            f'Stage-2 ignited at alt={alt/1000:.1f} km')

        elif self._phase == self.PHASE_STAGE_2_BURN:
            # Stage 2 exhausted → coast to apoapsis
            if (self.stage2._is_exhausted or self.stage2.propellant_fraction < 0.02):
                self.stage2.shutdown()
                self._current_stage_idx = 2
                self._phase = self.PHASE_COAST
                self._log_event('STAGE_2_CUTOFF',
                                f'Stage-2 MECO at alt={alt/1000:.1f} km, v={self.speed:.0f} m/s')

        elif self._phase == self.PHASE_COAST:
            # Check for circularisation opportunity (near apoapsis, altitude ≈ target)
            if alt >= TARGET_ALTITUDE * 0.98:
                self._phase = self.PHASE_CIRC_BURN
                self._log_event('CIRC_BURN_START',
                                f'Circularisation burn started at alt={alt/1000:.1f} km')

        elif self._phase == self.PHASE_CIRC_BURN:
            # No propellant on upper stage after MECO; this is a simulation-level
            # velocity adjustment to finalise circular orbit (represents residual
            # from stage-2 mixture ratio optimisation / ullage motors)
            elems = orbital_elements(pos, vel)
            ecc   = elems['eccentricity']
            if ecc < 0.005 and abs(alt - TARGET_ALTITUDE) < 5_000.0:
                self._deploy_cubesat()

        elif self._phase == self.PHASE_CUBESAT_SEP:
            self._phase = self.PHASE_COMPLETE

    # ── Circularisation (simulation-level adjustment) ─────────────────────────

    def _circularisation_adjustment(self, dt: float):
        """
        During CIRC_BURN phase, apply a small prograde velocity adjustment
        each step to circularise the orbit, representing residual stage-2 thrust.
        """
        pos = self._state[0:2]
        vel = self._state[2:4]
        r   = np.linalg.norm(pos)

        v_circ = orbital_velocity(r)
        v_now  = np.linalg.norm(vel)
        dv_needed = v_circ - v_now

        if abs(dv_needed) < 0.5:
            return   # Close enough

        # Apply prograde or retrograde nudge (max 20 m/s per second)
        max_dv_per_step = 15.0 * dt
        dv = np.clip(dv_needed, -max_dv_per_step, max_dv_per_step)

        if v_now > 0:
            v_hat = vel / v_now
            self._state[2:4] = vel + dv * v_hat

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, dt: float) -> dict:
        """
        Advance rocket simulation by dt seconds.

        Returns
        -------
        dict  Snapshot of vehicle state.
        """
        if self._phase == self.PHASE_PRELAUNCH:
            return self._snapshot(0.0, 0.0, 0.0)

        # ── Get active stage thrust ────────────────────────────────────────
        stage  = self._active_stage()
        thrust = 0.0

        if stage is not None and self._phase not in [
                self.PHASE_COAST, self.PHASE_CUBESAT_SEP,
                self.PHASE_COMPLETE, self.PHASE_STAGE_SEP_1]:
            throttle = self._compute_throttle()
            thrust   = stage.burn(dt, throttle)

        # ── Circularisation adjustment ────────────────────────────────────
        if self._phase == self.PHASE_CIRC_BURN:
            self._circularisation_adjustment(dt)

        # ── Guidance: update pitch ─────────────────────────────────────────
        if self._phase in [self.PHASE_GRAVITY_TURN,
                           self.PHASE_STAGE_2_BURN,
                           self.PHASE_CIRC_BURN]:
            self._pitch_angle = self._gravity_turn_pitch()
        elif self._phase in [self.PHASE_LIFTOFF, self.PHASE_VERTICAL]:
            self._pitch_angle = np.pi / 2.0   # straight up

        # ── Integrate dynamics ────────────────────────────────────────────
        def deriv(s):
            return self._derivatives(s, thrust, self._cross_section)

        self._state = runge_kutta_4(self._state, dt, deriv)

        # ── Drag magnitude (for telemetry) ────────────────────────────────
        alt = self.altitude
        drag_mag = np.linalg.norm(
            drag_force(self.velocity, alt, self._cross_section, 0.35)
        )

        # ── Acceleration felt ─────────────────────────────────────────────
        m     = self.total_mass()
        accel = thrust / m if m > 0 else 0.0

        # ── Phase state machine ───────────────────────────────────────────
        self._update_phase(thrust)

        self._time += dt

        # ── Record history ────────────────────────────────────────────────
        self._history_time.append(self._time)
        self._history_pos.append(self.position)
        self._history_vel.append(self.velocity)
        self._history_alt.append(self.altitude / 1000.0)
        self._history_speed.append(self.speed / 1000.0)
        self._history_accel.append(accel)
        self._history_mass.append(m)
        self._history_phase.append(self._phase)
        self._history_thrust.append(thrust)
        self._history_drag.append(drag_mag)

        return self._snapshot(thrust, drag_mag, accel)

    def _snapshot(self, thrust: float, drag: float, accel: float) -> dict:
        stage = self._active_stage()
        return {
            'time':           self._time,
            'phase':          self._phase,
            'position':       self.position,
            'velocity':       self.velocity,
            'altitude_m':     self.altitude,
            'altitude_km':    self.altitude / 1000.0,
            'speed_ms':       self.speed,
            'speed_kms':      self.speed / 1000.0,
            'mass_kg':        self.total_mass(),
            'thrust_N':       thrust,
            'drag_N':         drag,
            'accel_ms2':      accel,
            'pitch_deg':      np.degrees(self._pitch_angle),
            'stage_idx':      self._current_stage_idx,
            'prop_s1':        self.stage1.propellant_fraction,
            'prop_s2':        self.stage2.propellant_fraction,
            'fairing_off':    self._fairing_jettisoned,
            'cubesat_deployed': self._payload_deployed,
            'cubesat_pos':    self._cubesat_position,
            'cubesat_vel':    self._cubesat_velocity,
            'events':         self._events,
        }

    # ── Control ───────────────────────────────────────────────────────────────

    def ignite(self):
        """Commence launch sequence."""
        self.stage1.ignite()
        self._phase    = self.PHASE_LIFTOFF
        self._throttle = 1.0
        self._log_event('IGNITION', 'Main engine ignition — LIFTOFF')

    # ── History ───────────────────────────────────────────────────────────────

    @property
    def history(self) -> dict:
        return {
            'time':    self._history_time,
            'pos':     self._history_pos,
            'vel':     self._history_vel,
            'alt':     self._history_alt,
            'speed':   self._history_speed,
            'accel':   self._history_accel,
            'mass':    self._history_mass,
            'phase':   self._history_phase,
            'thrust':  self._history_thrust,
            'drag':    self._history_drag,
        }

    @property
    def events(self) -> List[MissionEvent]:
        return self._events