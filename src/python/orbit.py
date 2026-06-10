"""
orbit.py - Orbit determination, propagation, and manoeuvre planning.

Provides:
  • OrbitalState  – snapshot of position, velocity, and derived elements
  • OrbitPropagator – Kepler + J2 perturbation propagator (for reference orbits)
  • ManoeuvreController – closed-loop orbit correction logic
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from physics import (
    R_EARTH, MU_EARTH, TARGET_RADIUS, TARGET_ALTITUDE,
    orbital_velocity, orbital_elements,
    runge_kutta_4, gravity_acceleration, drag_force,
    atmospheric_density,
)


# J2 oblateness coefficient (Earth)
J2 = 1.08262668e-3


@dataclass
class OrbitalState:
    """
    Snapshot of spacecraft orbital state.
    """
    time:     float
    position: np.ndarray    # [m]
    velocity: np.ndarray    # [m/s]
    mass:     float         # [kg]

    # Derived elements (populated lazily)
    _elements: Optional[dict] = None

    def elements(self) -> dict:
        if self._elements is None:
            self._elements = orbital_elements(self.position, self.velocity)
        return self._elements

    @property
    def altitude(self) -> float:
        return np.linalg.norm(self.position) - R_EARTH

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)

    @property
    def radius(self) -> float:
        return np.linalg.norm(self.position)

    def deviation_from_target(self) -> dict:
        """Deviation of current orbit from target circular orbit."""
        elems = self.elements()
        target_alt   = TARGET_ALTITUDE
        target_r     = TARGET_RADIUS
        target_v     = orbital_velocity(target_r)
        target_period= 2.0 * np.pi * np.sqrt(target_r**3 / MU_EARTH)

        alt_dev   = self.altitude           - target_alt
        sma_dev   = elems['semi_major_axis']- target_r
        ecc_dev   = elems['eccentricity']   - 0.0        # circular target
        speed_dev = self.speed              - target_v

        return {
            'altitude_deviation_m':    alt_dev,
            'sma_deviation_m':         sma_dev,
            'eccentricity_deviation':  ecc_dev,
            'speed_deviation_ms':      speed_dev,
            'target_altitude_m':       target_alt,
            'target_speed_ms':         target_v,
            'target_period_s':         target_period,
        }


class OrbitPropagator:
    """
    High-fidelity numerical propagator.

    Includes:
      • Two-body gravity (Keplerian)
      • J2 oblateness perturbation (in 2-D: simplified secular drift)
      • Atmospheric drag (exponential model)
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray, mass: float):
        self._state = np.concatenate([position, velocity])  # [x, y, vx, vy]
        self._mass  = mass
        self._time  = 0.0
        self._history: List[OrbitalState] = []
        self._cross_section = 0.02   # m²  CubeSat 6U face-on

    def _derivatives(self, state: np.ndarray, mass: float) -> np.ndarray:
        pos = state[0:2]
        vel = state[2:4]
        r   = np.linalg.norm(pos)

        # Gravitational acceleration (point mass)
        a_grav = gravity_acceleration(pos)

        # J2 perturbation (in 2-D equatorial plane, secular approximation)
        # Full 3-D J2 simplifies in equatorial 2-D to a radial perturbation
        r5 = r**5
        j2_factor = 1.5 * J2 * MU_EARTH * (R_EARTH**2) / r5
        # In-plane equatorial J2 (radial component only)
        a_j2 = j2_factor * pos * (-1.0)   # always slightly inward for equatorial orbit

        # Atmospheric drag
        altitude_m = r - R_EARTH
        a_drag = drag_force(vel, altitude_m,
                            self._cross_section, 2.2) / mass

        a_total = a_grav + a_j2 + a_drag
        return np.array([vel[0], vel[1], a_total[0], a_total[1]])

    def step(self, dt: float) -> OrbitalState:
        """Advance propagator by dt seconds."""
        def deriv(s):
            return self._derivatives(s, self._mass)

        self._state = runge_kutta_4(self._state, dt, deriv)
        self._time += dt

        state = OrbitalState(
            time     = self._time,
            position = self._state[0:2].copy(),
            velocity = self._state[2:4].copy(),
            mass     = self._mass,
        )
        self._history.append(state)
        return state

    def propagate(self, duration_s: float, dt: float = 10.0) -> List[OrbitalState]:
        """Propagate forward in time and return state history."""
        states = []
        t = 0.0
        while t < duration_s:
            step_dt = min(dt, duration_s - t)
            s = self.step(step_dt)
            states.append(s)
            t += step_dt
        return states

    @property
    def position(self) -> np.ndarray:
        return self._state[0:2]

    @property
    def velocity(self) -> np.ndarray:
        return self._state[2:4]

    @property
    def history(self) -> List[OrbitalState]:
        return self._history


class ManoeuvreController:
    """
    Closed-loop orbit maintenance / correction controller.

    Uses the following strategy:
    1. Monitor eccentricity and altitude errors.
    2. If eccentricity > threshold → circularisation burn (apoapsis raise/lower).
    3. If altitude error > threshold → Hohmann correction.
    4. Implements proportional throttle control.
    """

    ALT_DEADBAND_M   = 2_000.0    # ±2 km altitude band
    ECC_DEADBAND     = 0.003       # eccentricity tolerance
    MAX_BURN_DV_MS   = 50.0        # max correction Δv per manoeuvre [m/s]

    def __init__(self):
        self._active:         bool  = False
        self._correction_dv:  float = 0.0    # remaining Δv to burn [m/s]
        self._burn_direction: np.ndarray = np.zeros(2)
        self._phase:          str   = 'IDLE'

        self._total_correction_dv: float = 0.0
        self._manoeuvre_count:     int   = 0

    def assess(self,
               position: np.ndarray,
               velocity: np.ndarray) -> Tuple[bool, str, float]:
        """
        Check orbit status and determine if correction is needed.

        Returns
        -------
        (needs_correction, reason, delta_v_needed)
        """
        elems    = orbital_elements(position, velocity)
        altitude = elems['altitude']
        ecc      = elems['eccentricity']
        sma      = elems['semi_major_axis']
        v_circ   = orbital_velocity(TARGET_RADIUS)
        v_now    = np.linalg.norm(velocity)

        alt_error = altitude - TARGET_ALTITUDE
        ecc_error = ecc

        if abs(alt_error) > self.ALT_DEADBAND_M:
            dv_needed = abs(v_circ - v_now)
            dv_needed = min(dv_needed, self.MAX_BURN_DV_MS)
            return True, f"ALT_ERR {alt_error/1000:.1f} km", dv_needed

        if ecc_error > self.ECC_DEADBAND:
            # Circularisation: burn at apoapsis (or periapsis)
            # Δv ≈ ecc * v_circ / 2 (approximation)
            dv_needed = ecc_error * v_circ * 0.5
            dv_needed = min(dv_needed, self.MAX_BURN_DV_MS)
            return True, f"ECC_ERR {ecc_error:.4f}", dv_needed

        return False, "NOMINAL", 0.0

    def initiate_correction(self,
                            position: np.ndarray,
                            velocity: np.ndarray,
                            dv_needed: float):
        """
        Start a correction burn.
        Direction is prograde or retrograde depending on sign of altitude error.
        """
        elems     = orbital_elements(position, velocity)
        alt_error = elems['altitude'] - TARGET_ALTITUDE

        v_mag = np.linalg.norm(velocity)
        if v_mag > 0:
            prograde = velocity / v_mag
        else:
            prograde = np.array([0.0, 1.0])

        # If too low → prograde burn to raise orbit
        # If too high → retrograde burn to lower orbit
        if alt_error < 0:
            self._burn_direction = prograde
        else:
            self._burn_direction = -prograde

        self._correction_dv = dv_needed
        self._active        = True
        self._phase         = 'CORRECTING'
        self._manoeuvre_count += 1

    def get_thrust_direction(self) -> np.ndarray:
        """Return unit vector for current correction thrust."""
        return self._burn_direction.copy()

    def consume_dv(self, dv_applied: float):
        """Called each step to decrement remaining correction Δv."""
        self._correction_dv        -= abs(dv_applied)
        self._total_correction_dv  += abs(dv_applied)
        if self._correction_dv <= 0.0:
            self._active        = False
            self._correction_dv = 0.0
            self._phase         = 'IDLE'

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def remaining_dv(self) -> float:
        return max(0.0, self._correction_dv)

    @property
    def phase(self) -> str:
        return self._phase

    def get_telemetry(self) -> dict:
        return {
            'active':               self._active,
            'remaining_dv_ms':      self._correction_dv,
            'total_correction_dv':  self._total_correction_dv,
            'manoeuvre_count':      self._manoeuvre_count,
            'phase':                self._phase,
        }