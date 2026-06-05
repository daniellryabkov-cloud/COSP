# Scientific CubeSat Launch Simulator

GNU Octave source code for a physically based CubeSat launch simulation.  The default case launches a 1U CubeSat-class payload from Earth's surface on an educational three-stage rocket, performs finite-burn ascent with staging and mass depletion, deploys the CubeSat near orbital insertion, then propagates the spacecraft in Low Earth Orbit.

## Run

Use the ASCII copy if Octave has trouble with the original Cyrillic desktop path:

```octave
cd 'C:/tmp/COSP_SIMULATION'
[history, mission, C] = main();
```

Without plots:

```octave
[history, mission, C] = main(false);
```

Export MP4 animation if `VideoWriter` is available in your Octave installation:

```octave
params.export_mp4 = true;
params.animation_filename = 'launch.mp4';
[history, mission, C] = main(true, params);
```

The simulator saves `cubesat_launch_output.mat`.

## Files

- `constants.m` - physical constants and CubeSat reference properties
- `vector_utils.m` - vector, angle, and rotation helpers
- `gravity_model.m` - Newtonian point-mass Earth gravity
- `atmosphere_model.m` - standard atmosphere density, pressure, temperature
- `rocket_model.m` - launch site, multi-stage vehicle, payload, guidance setup
- `rk4_propagator.m` - RK4 integration for rocket ascent and satellite coast
- `orbital_elements.m` - classical orbital element calculation
- `ground_track.m` - ECI to rotating-Earth latitude/longitude conversion
- `visualization.m` - 3D plots, ground track, animation, MP4 export
- `main.m` - mission entry point

## Main Equations

Newtonian gravity:

```text
a_g = -mu r / |r|^3
```

Rocket mass depletion:

```text
dm/dt = -T / (Isp g0)
```

Rocket translational dynamics:

```text
dr/dt = v
dv/dt = a_g + (F_thrust + F_drag) / m
```

Atmospheric drag:

```text
v_rel = v_inertial - omega_earth x r
F_drag = -0.5 rho |v_rel|^2 Cd A v_rel_hat
```

Specific orbital energy:

```text
epsilon = |v|^2 / 2 - mu / |r|
```

Specific angular momentum:

```text
h = r x v
```

Eccentricity vector:

```text
e_vec = (v x h) / mu - r / |r|
```

Semi-major axis:

```text
a = -mu / (2 epsilon)
```

Elliptic orbital period:

```text
T_orbit = 2 pi sqrt(a^3 / mu)
```

Ground track:

```text
theta = theta0 + omega_earth t
r_ecef = R_eci_to_ecef(theta) r_eci
latitude = asin(z_ecef / |r_ecef|)
longitude = atan2(y_ecef, x_ecef)
```

## Assumptions

- SI units everywhere.
- Earth gravity is two-body point-mass gravity.
- Earth is spherical for altitude, drag altitude, and ground track.
- Atmosphere corotates with Earth.
- Atmosphere follows a 1976 Standard Atmosphere layer model to 84.852 km with an exponential high-altitude extension.
- Thrust is vacuum-equivalent and constant within each stage.
- Staging is instantaneous at propellant depletion.
- The CubeSat is deployed after final-stage burnout with a small prograde impulse.
- The default ascent guidance is an explicit gravity-turn pitch schedule, not closed-loop optimal guidance.

## Limitations

- No J2 or higher-order geopotential perturbations.
- No third-body gravity.
- No solar radiation pressure.
- No winds, weather, or launch rail dynamics.
- No engine throttling transients, mixture-ratio effects, or pressure-dependent thrust.
- No structural loads, heating, attitude dynamics, telemetry, power, communications, or thermal subsystems.
- The default vehicle is educational, not a certified real launch vehicle.
- RK4 is accurate for this short simulation but is not symplectic for very long-term orbit propagation.
