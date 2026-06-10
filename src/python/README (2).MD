# CubeSat LEO Insertion Mission Simulator

A scientific aerospace simulation that models the complete deployment of a 6U CubeSat into a 500 km circular Low Earth Orbit (LEO).

The project combines launch vehicle dynamics, atmospheric flight, orbital mechanics, spacecraft subsystem simulation, and mission visualization within a MATLAB-inspired engineering environment.

Designed as part of the COSP (CubeSat Orbital Simulation Platform) project, the simulator reproduces the full mission timeline from liftoff through orbital operations while maintaining physically-based models and engineering-oriented visualization.

---

## Mission Overview

The simulator reproduces the following mission sequence:

1. Liftoff and vertical ascent
2. Gravity turn maneuver
3. First-stage powered flight
4. Maximum dynamic pressure (Max-Q) management
5. Stage separation
6. Second-stage orbital insertion burn
7. Payload fairing jettison
8. Coast phase to apoapsis
9. Circularization burn
10. CubeSat deployment
11. Spacecraft stabilization
12. Nominal mission operations

---

## Core Features

* Two-stage launch vehicle simulation
* Realistic atmospheric ascent
* Orbital insertion and circularization
* CubeSat deployment mechanics
* Spacecraft power system simulation
* Battery charge/discharge modelling
* Solar panel power generation
* Eclipse detection
* Cold-gas orbit correction system
* Telemetry logging
* Mission event tracking
* Scientific mission visualization
* GIF and MP4 export
* Automated mission analysis plots

---

## Physics Models

The simulation uses classical aerospace engineering models:

| Model                 | Description                               |
| --------------------- | ----------------------------------------- |
| Gravity               | Newtonian point-mass gravity              |
| Atmosphere            | US Standard Atmosphere 1976               |
| Aerodynamics          | Velocity-dependent drag model             |
| Propulsion            | Thrust and mass-flow equations            |
| Orbital Mechanics     | Vis-viva equation and eccentricity vector |
| Numerical Integration | Fourth-order Runge-Kutta (RK4)            |

All vehicle states are propagated numerically throughout the mission.
