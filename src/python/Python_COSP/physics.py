"""
physics.py - Physical constants, gravitational and atmospheric models,
             and core force/acceleration calculations.
"""

import numpy as np

# ── Universal constants ────────────────────────────────────────────────────────
G          = 6.67430e-11        # Gravitational constant  [m³ kg⁻¹ s⁻²]
M_EARTH    = 5.97219e24         # Earth mass              [kg]
R_EARTH    = 6_371_000.0        # Earth mean radius       [m]
MU_EARTH   = G * M_EARTH        # Standard gravitational parameter [m³ s⁻²]
OMEGA_EARTH = 7.2921150e-5      # Earth rotation rate     [rad/s]

# Atmosphere (US Standard Atmosphere 1976, simplified)
SEA_LEVEL_DENSITY = 1.225       # [kg/m³]
SCALE_HEIGHT      = 8_500.0     # Density scale height    [m]
SEA_LEVEL_PRESSURE = 101_325.0  # [Pa]
SEA_LEVEL_TEMP    = 288.15      # [K]
LAPSE_RATE        = 0.0065      # [K/m] (troposphere)

# Drag
CD_ROCKET    = 0.35             # Drag coefficient (cylindrical rocket)
CD_CUBESAT   = 2.20             # Drag coefficient (tumbling CubeSat, worst case)

# Target orbit
TARGET_ALTITUDE = 500_000.0     # [m]   500 km LEO
TARGET_RADIUS   = R_EARTH + TARGET_ALTITUDE


def gravity_acceleration(position: np.ndarray) -> np.ndarray:
    """
    Newtonian point-mass gravitational acceleration vector.

    Parameters
    ----------
    position : np.ndarray, shape (2,)
        Position vector [x, y] in metres (ECI-like 2-D frame).

    Returns
    -------
    np.ndarray, shape (2,)
        Acceleration [m/s²].
    """
    r = np.linalg.norm(position)
    if r < 1.0:
        return np.zeros(2)
    return -MU_EARTH / r**3 * position


def atmospheric_density(altitude_m: float) -> float:
    """
    Exponential atmosphere model.

    Parameters
    ----------
    altitude_m : float
        Geometric altitude above mean sea level [m].

    Returns
    -------
    float
        Air density [kg/m³].  Returns 0 above 600 km.
    """
    if altitude_m > 600_000.0:
        return 0.0
    if altitude_m < 0.0:
        altitude_m = 0.0
    return SEA_LEVEL_DENSITY * np.exp(-altitude_m / SCALE_HEIGHT)


def atmospheric_pressure(altitude_m: float) -> float:
    """Barometric formula (troposphere + rough upper-atmosphere)."""
    if altitude_m < 0:
        altitude_m = 0.0
    if altitude_m <= 11_000.0:
        T = SEA_LEVEL_TEMP - LAPSE_RATE * altitude_m
        return SEA_LEVEL_PRESSURE * (T / SEA_LEVEL_TEMP) ** 5.2561
    # Simplified isothermal above tropopause
    p11 = atmospheric_pressure(11_000.0)
    T11 = SEA_LEVEL_TEMP - LAPSE_RATE * 11_000.0
    return p11 * np.exp(-(altitude_m - 11_000.0) / (287.05 * T11 / 9.80665))


def drag_force(velocity: np.ndarray,
               altitude_m: float,
               cross_section_area: float,
               cd: float) -> np.ndarray:
    """
    Aerodynamic drag force vector (opposes velocity).

    F_drag = -½ ρ v² C_d A  v̂

    Parameters
    ----------
    velocity        : np.ndarray [m/s]
    altitude_m      : float      [m]
    cross_section_area : float   [m²]
    cd              : float      dimensionless drag coefficient

    Returns
    -------
    np.ndarray [N]
    """
    rho = atmospheric_density(altitude_m)
    v_mag = np.linalg.norm(velocity)
    if v_mag < 1e-6 or rho < 1e-20:
        return np.zeros(2)
    v_hat = velocity / v_mag
    f_mag = 0.5 * rho * v_mag**2 * cd * cross_section_area
    return -f_mag * v_hat


def orbital_velocity(radius_m: float) -> float:
    """
    Circular orbital speed at a given orbital radius.

    v_circ = sqrt(μ / r)

    Parameters
    ----------
    radius_m : float   Distance from Earth centre [m].

    Returns
    -------
    float  Speed [m/s].
    """
    return np.sqrt(MU_EARTH / radius_m)


def specific_orbital_energy(position: np.ndarray,
                             velocity: np.ndarray) -> float:
    """
    Specific mechanical energy  ε = v²/2 − μ/r  [J/kg].
    """
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)
    return 0.5 * v**2 - MU_EARTH / r


def semi_major_axis(position: np.ndarray, velocity: np.ndarray) -> float:
    """
    Semi-major axis from vis-viva equation.

    a = −μ / (2ε)

    Returns metres (positive for elliptical orbits).
    """
    eps = specific_orbital_energy(position, velocity)
    if eps >= 0:
        return float('inf')   # hyperbolic / parabolic
    return -MU_EARTH / (2.0 * eps)


def eccentricity_vector(position: np.ndarray,
                         velocity: np.ndarray) -> np.ndarray:
    """
    Laplace–Runge–Lenz eccentricity vector.

    e⃗ = (v²/μ − 1/r) r⃗ − (r⃗·v⃗)/μ  v⃗
    """
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)
    rdotv = np.dot(position, velocity)
    return ((v**2 / MU_EARTH - 1.0 / r) * position
            - (rdotv / MU_EARTH) * velocity)


def orbital_elements(position: np.ndarray,
                     velocity: np.ndarray) -> dict:
    """
    Compute classical 2-D orbital elements.

    Returns
    -------
    dict with keys:
        semi_major_axis  [m]
        eccentricity     [-]
        periapsis        [m]  altitude above surface
        apoapsis         [m]  altitude above surface
        orbital_period   [s]
        altitude         [m]  current altitude
    """
    r = np.linalg.norm(position)
    altitude = r - R_EARTH

    a = semi_major_axis(position, velocity)
    e_vec = eccentricity_vector(position, velocity)
    e = np.linalg.norm(e_vec)
    e = min(e, 0.9999)   # clamp for display

    periapsis = a * (1.0 - e) - R_EARTH
    apoapsis  = a * (1.0 + e) - R_EARTH

    if a > 0:
        period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
    else:
        period = float('inf')

    return {
        'semi_major_axis': a,
        'eccentricity':    e,
        'periapsis':       periapsis,
        'apoapsis':        apoapsis,
        'orbital_period':  period,
        'altitude':        altitude,
    }


def hohmann_delta_v(r1: float, r2: float) -> tuple:
    """
    Hohmann transfer Δv pair between two circular orbits.

    Parameters
    ----------
    r1, r2 : float   Orbital radii [m].

    Returns
    -------
    (dv1, dv2)  both in m/s  (positive = prograde burn)
    """
    v1     = orbital_velocity(r1)
    v2     = orbital_velocity(r2)
    v_t1   = np.sqrt(MU_EARTH * (2.0 / r1 - 2.0 / (r1 + r2)))
    v_t2   = np.sqrt(MU_EARTH * (2.0 / r2 - 2.0 / (r1 + r2)))
    dv1    = v_t1 - v1
    dv2    = v2   - v_t2
    return dv1, dv2


def runge_kutta_4(state: np.ndarray,
                  dt: float,
                  derivatives_fn) -> np.ndarray:
    """
    4th-order Runge–Kutta integrator.

    Parameters
    ----------
    state          : np.ndarray   Current state vector.
    dt             : float        Time step [s].
    derivatives_fn : callable     f(state) -> d(state)/dt

    Returns
    -------
    np.ndarray  New state after one step.
    """
    k1 = derivatives_fn(state)
    k2 = derivatives_fn(state + 0.5 * dt * k1)
    k3 = derivatives_fn(state + 0.5 * dt * k2)
    k4 = derivatives_fn(state +       dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)