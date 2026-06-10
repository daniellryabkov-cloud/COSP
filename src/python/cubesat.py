"""
cubesat.py - 6U CubeSat spacecraft model.

Integrates:
  • On-board propulsion (cold-gas thrusters for orbit correction)
  • Power system (solar panels + battery)
  • Camera payload
  • Telemetry system
  • Attitude control (simplified nadir-pointing)
  • Orbit correction controller
"""

import numpy as np
from typing import Optional

from physics import (
    R_EARTH, MU_EARTH, TARGET_RADIUS, TARGET_ALTITUDE,
    gravity_acceleration, drag_force, orbital_elements,
    orbital_velocity, runge_kutta_4,
)
from power     import PowerSystem
from camera    import Camera
from telemetry import TelemetrySystem
from orbit     import ManoeuvreController, OrbitalState


# ── CubeSat physical parameters ───────────────────────────────────────────────

CUBESAT_DRY_MASS        = 8.0       # kg   (6U, fully loaded)
CUBESAT_PROP_MASS_INIT  = 0.5       # kg   cold-gas N2 propellant
CUBESAT_CROSS_SECTION   = 0.02      # m²   (6U face-on 10×20 cm)
CUBESAT_CD              = 2.2       # drag coefficient (conservative)

# Cold-gas thruster parameters
THRUSTER_THRUST         = 0.5       # N
THRUSTER_ISP            = 60.0      # s  (N2 cold gas)
THRUSTER_MDOT           = THRUSTER_THRUST / (THRUSTER_ISP * 9.80665)   # kg/s


class CubeSat:
    """
    Full 6U CubeSat spacecraft simulation.

    State vector: [x, y, vx, vy]  in ECI-like 2-D frame.

    Mission phases managed here:
      SEPARATION       → just released from launch vehicle
      STABILIZATION    → attitude acquisition, orbit check
      NOMINAL_MISSION  → imaging, downlink, science operations
      CORRECTING       → orbit correction burn in progress
      SAFE_MODE        → battery low, minimum power mode
    """

    PHASES = {
        'SEPARATION':    0,
        'STABILIZATION': 1,
        'NOMINAL_MISSION': 2,
        'CORRECTING':    3,
        'SAFE_MODE':     4,
    }

    STAB_DURATION = 120.0       # seconds in stabilisation before going nominal

    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 separation_time: float):
        # Validate insertion orbit before accepting
        self._state = np.array([
            position[0], position[1],
            velocity[0], velocity[1],
        ], dtype=float)

        self._dry_mass        = CUBESAT_DRY_MASS
        self._prop_mass       = CUBESAT_PROP_MASS_INIT
        self._separation_time = separation_time
        self._mission_time    = separation_time
        self._phase           = 'SEPARATION'
        self._phase_start     = separation_time

        # Attitude (simplified: angle from +X toward +Y, nadir-pointing)
        self._attitude        = np.arctan2(position[1], position[0]) + np.pi/2

        # Subsystems
        self.power    = PowerSystem()
        self.camera   = Camera()
        self.telemetry= TelemetrySystem()
        self.manoeuvre= ManoeuvreController()

        # Flags
        self._panels_deployed = False
        self._imaging_active  = False

        # History for plots
        self._history_time:    list = []
        self._history_alt:     list = []
        self._history_speed:   list = []
        self._history_ecc:     list = []
        self._history_soc:     list = []
        self._history_phase:   list = []
        self._history_pos:     list = []

        self.telemetry.log_event(separation_time, 'SEPARATION',
                                 'CubeSat separated from launch vehicle upper stage')

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self._state[0:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._state[2:4].copy()

    @property
    def mass(self) -> float:
        return self._dry_mass + self._prop_mass

    @property
    def altitude(self) -> float:
        return np.linalg.norm(self._state[0:2]) - R_EARTH

    @property
    def speed(self) -> float:
        return np.linalg.norm(self._state[2:4])

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def mission_time(self) -> float:
        return self._mission_time

    # ── Dynamics ──────────────────────────────────────────────────────────────

    def _thruster_acceleration(self, direction: np.ndarray, throttle: float = 1.0) -> np.ndarray:
        """Acceleration from cold-gas thruster."""
        if self._prop_mass <= 0:
            return np.zeros(2)
        thrust_N = THRUSTER_THRUST * throttle
        return thrust_N / self.mass * direction

    def _derivatives(self, state: np.ndarray, extra_accel: np.ndarray) -> np.ndarray:
        pos = state[0:2]
        vel = state[2:4]
        alt = np.linalg.norm(pos) - R_EARTH

        a_grav = gravity_acceleration(pos)
        a_drag = drag_force(vel, alt,
                            CUBESAT_CROSS_SECTION,
                            CUBESAT_CD) / self.mass
        a_extra = extra_accel

        a_total = a_grav + a_drag + a_extra
        return np.array([vel[0], vel[1], a_total[0], a_total[1]])

    # ── Phase management ──────────────────────────────────────────────────────

    def _transition_phase(self, new_phase: str):
        if new_phase != self._phase:
            self.telemetry.log_event(
                self._mission_time,
                'PHASE_CHANGE',
                f'{self._phase} → {new_phase}',
            )
            self._phase = new_phase
            self._phase_start = self._mission_time

    def _update_phase_logic(self):
        """State machine for mission phase transitions."""
        phase = self._phase
        dt_in_phase = self._mission_time - self._phase_start

        if phase == 'SEPARATION':
            # Deploy panels immediately
            if not self._panels_deployed:
                self._panels_deployed = True
                self.telemetry.log_event(self._mission_time, 'DEPLOY',
                                          'Solar panels deployed')
            if dt_in_phase > 30.0:
                self._transition_phase('STABILIZATION')

        elif phase == 'STABILIZATION':
            if dt_in_phase > self.STAB_DURATION:
                # Check orbit
                needs_corr, reason, dv = self.manoeuvre.assess(
                    self.position, self.velocity)
                if needs_corr:
                    self.manoeuvre.initiate_correction(
                        self.position, self.velocity, dv)
                    self._transition_phase('CORRECTING')
                    self.telemetry.log_event(
                        self._mission_time, 'MANOEUVRE',
                        f'Orbit correction initiated: {reason}  Δv={dv:.2f} m/s')
                else:
                    self._transition_phase('NOMINAL_MISSION')
                    self.camera.start_imaging()
                    self._imaging_active = True
                    self.power.enable_subsystem('camera')
                    self.telemetry.log_event(self._mission_time, 'MISSION',
                                             'Nominal mission phase started — imaging active')

        elif phase == 'NOMINAL_MISSION':
            # Periodically assess orbit health
            if dt_in_phase > 0 and dt_in_phase % 300.0 < 5.0:   # every ~5 min
                needs_corr, reason, dv = self.manoeuvre.assess(
                    self.position, self.velocity)
                if needs_corr:
                    self.manoeuvre.initiate_correction(
                        self.position, self.velocity, dv)
                    self._transition_phase('CORRECTING')
                    self.power.enable_subsystem('propulsion')
                    self.camera.stop_imaging()
                    self.power.disable_subsystem('camera')

            # Safe mode check
            if self.power.battery.is_critically_low():
                self._transition_phase('SAFE_MODE')
                self.camera.stop_imaging()
                self.power.disable_subsystem('camera')
                self.power.disable_subsystem('radio_tx')

        elif phase == 'CORRECTING':
            if not self.manoeuvre.is_active:
                self._transition_phase('STABILIZATION')
                self._phase_start = self._mission_time - self.STAB_DURATION + 10.0
                self.power.disable_subsystem('propulsion')

        elif phase == 'SAFE_MODE':
            if self.power.battery.state_of_charge > 0.40:
                self._transition_phase('STABILIZATION')
                self.power.enable_subsystem('radio_tx')
                self.telemetry.log_event(self._mission_time, 'RECOVERY',
                                          'Exiting safe mode — battery recovered')

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, dt: float) -> dict:
        """
        Advance CubeSat simulation by dt seconds.

        Returns
        -------
        dict  Full spacecraft state snapshot.
        """
        self._mission_time += dt

        # Phase state machine
        self._update_phase_logic()

        # Correction thrust
        extra_accel = np.zeros(2)
        dv_this_step = 0.0

        if self._phase == 'CORRECTING' and self.manoeuvre.is_active:
            direction = self.manoeuvre.get_thrust_direction()
            throttle  = 1.0
            a_thrust  = self._thruster_acceleration(direction, throttle)
            extra_accel = a_thrust

            # Propellant consumed
            dm = THRUSTER_MDOT * throttle * dt
            dm = min(dm, self._prop_mass)
            self._prop_mass = max(0.0, self._prop_mass - dm)

            # Track Δv consumed (F=ma → a*dt = Δv)
            dv_this_step = np.linalg.norm(a_thrust) * dt
            self.manoeuvre.consume_dv(dv_this_step)

        # Numerical integration (RK4)
        def deriv(s):
            return self._derivatives(s, extra_accel)

        self._state = runge_kutta_4(self._state, dt, deriv)

        # Attitude update (nadir-pointing: normal toward Earth centre)
        r_vec = self._state[0:2]
        self._attitude = np.arctan2(r_vec[1], r_vec[0]) + np.pi

        # Sun angle (simplified: Sun along +X, rotates slowly)
        sun_angle = 2.0 * np.pi * (self._mission_time / 31_557_600.0)

        # Power subsystem
        power_state = self.power.update(
            dt           = dt,
            mission_time = self._mission_time,
            position     = self.position,
            sun_angle    = sun_angle,
        )

        # Camera
        frame = self.camera.update(
            mission_time = self._mission_time,
            position     = self.position,
            altitude_m   = self.altitude,
            in_shadow    = power_state['in_shadow'],
        )

        # Orbital elements
        elems = orbital_elements(self.position, self.velocity)

        # Telemetry downlink simulation
        self.telemetry.update_downlink(self._mission_time, dt)

        # Collect telemetry packet (every PACKET_INTERVAL_S)
        pkt = self.telemetry.collect(
            mission_time     = self._mission_time,
            position         = self.position,
            velocity         = self.velocity,
            orbital_elements = elems,
            power_state      = power_state,
            phase            = self._phase,
            mass_kg          = self.mass,
            attitude         = self._attitude,
            imaging          = self._imaging_active,
        )

        # History
        self._history_time.append(self._mission_time)
        self._history_alt.append(self.altitude / 1000.0)   # km
        self._history_speed.append(self.speed  / 1000.0)   # km/s
        self._history_ecc.append(elems['eccentricity'])
        self._history_soc.append(power_state['battery_soc'] * 100.0)
        self._history_phase.append(self._phase)
        self._history_pos.append(self.position.copy())

        return {
            'mission_time':   self._mission_time,
            'phase':          self._phase,
            'position':       self.position,
            'velocity':       self.velocity,
            'mass':           self.mass,
            'altitude_m':     self.altitude,
            'altitude_km':    self.altitude / 1000.0,
            'speed_ms':       self.speed,
            'speed_kms':      self.speed / 1000.0,
            'elements':       elems,
            'power':          power_state,
            'camera':         self.camera.get_state(),
            'manoeuvre':      self.manoeuvre.get_telemetry(),
            'attitude_rad':   self._attitude,
            'prop_remaining': self._prop_mass,
        }

    # ── History accessors ─────────────────────────────────────────────────────

    @property
    def history(self) -> dict:
        return {
            'time':      self._history_time,
            'altitude':  self._history_alt,
            'speed':     self._history_speed,
            'ecc':       self._history_ecc,
            'soc':       self._history_soc,
            'phase':     self._history_phase,
            'positions': self._history_pos,
        }