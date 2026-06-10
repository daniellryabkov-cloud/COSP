COSP Project — Python Engineering Simulations

This repository contains Python-based tools and simulations for aerospace and robotics projects. The goal is to provide physically accurate models, scientific visualization, and reusable code for mission analysis, control system design, and CubeSat simulations. The repository includes experimental scripts, prototypes, and production-ready modules.

Key Features:

Written in Python 3.12+
Uses numpy, matplotlib, and pillow for computation and visualization
Supports both experimental and finalized simulation modules
Structured for easy extension and reuse
Includes archival files for past experiments and prototypes
CubeSat LEO Insertion Mission Simulator

A physically accurate 2D simulation of a two-stage rocket launch and CubeSat deployment into a 500 km circular low Earth orbit (LEO), designed in the style of MATLAB / NASA mission analysis tools.

Features
Full mission simulation from liftoff to on-orbit operations:
Stage 1 burn with gravity turn and Max-Q throttle
Stage 1 separation and Stage 2 ignition
Fairing jettison above 100 km
Main Engine Cutoff (MECO) at 97% of orbital speed
Coast to apoapsis
Circularization burn at 500 km
CubeSat deployment
On-orbit stabilization, nominal mission, orbit correction, and safe mode
Physics and integration:
4th-order Runge–Kutta integration
Newtonian gravity, US Standard Atmosphere 1976, drag, thrust
Adaptive timestep: 0.25 s dense / 8 s coast
CubeSat subsystems: solar panels, Li-ion battery, cold-gas thrusters, imaging camera, telemetry
Requirements
pip install numpy matplotlib pillow

For MP4 export, install ffmpeg:

# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
winget install ffmpeg
Run
python launch_simulation.py

Runtime: ~30–60 seconds depending on hardware.

Output
File	Description
mission_scientific.gif	Animated mission (444 frames, 15 fps)
mission_scientific.mp4	H.264 video (requires ffmpeg)
mission_analysis.png	10-panel static analysis chart
telemetry_log.json	Full event log and final orbit state
Mission Parameters
Parameter	Value
Target orbit	500 km circular LEO
Orbital velocity	7.616 km/s
Orbital period	~94.7 min
Launch site	Equatorial (0°, 0°)
Stage 1 thrust	560 kN vacuum
Stage 1 Isp	282 s
Stage 2 thrust	95 kN vacuum
Stage 2 Isp	348 s
CubeSat mass	8 kg (6U)
CubeSat propellant	0.5 kg cold-gas N₂
Battery capacity	20 Wh
Visualization
Scientific engineering style — no decorative effects
White background, muted engineering color palette
Smooth camera zoom (launch pad → orbit)
Live strip charts: altitude, velocity, battery SOC, eccentricity
Phase labels in monospace font
Scale bar in all zoom levels
File Structure
launch_simulation.py   — complete self-contained simulation (~950 lines)
├── physics helpers   gravity, drag, atmosphere, RK4, orbital elements
├── class Rocket      two-stage launch vehicle, guidance, staging
├── class CubeSat     on-orbit operations, power, GNC, camera
├── build_animation() matplotlib FuncAnimation → GIF / MP4
└── save_analysis_plots() 10-panel static figure
Mission Phases
Launch vehicle: VERTICAL → S1_BURN → S1_SEP → S2_BURN → COAST → CIRC_BURN → DEPLOY
CubeSat: SEPARATION → STABILIZATION → NOMINAL_MISSION
CORRECTING (orbit trim)
SAFE_MODE (low battery)
License

MIT
