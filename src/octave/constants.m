function C = constants()
%CONSTANTS Physical and numerical constants for the CubeSat launch mission.
%
% All quantities use SI units unless a field name explicitly says otherwise.
% The constants are collected in one structure so that future models
% (atmospheric drag, J2, solar radiation pressure, attitude, telemetry, and
% power) can share a single source of physical truth.

  % Earth's standard gravitational parameter, mu = G * M_earth.
  % Units: m^3/s^2.  This parameter appears directly in Newtonian gravity:
  % a = -mu * r / |r|^3.
  C.mu = 3.986004418e14;

  % Earth's equatorial radius.  Units: m.  Used for altitude and plotting.
  C.Re = 6378137.0;

  % Earth's sidereal rotation rate.  Units: rad/s.  Used to convert inertial
  % ECI position vectors into Earth-fixed longitude for the ground track.
  C.omega_earth = 7.2921150e-5;

  % Standard gravitational acceleration at sea level.  Units: m/s^2.
  % Not used in orbital propagation, but useful for subsystem expansion.
  C.g0 = 9.80665;

  % International Standard Atmosphere constants for the troposphere through
  % lower thermosphere approximation used in atmosphere_model.m.
  C.atm.R_air = 287.05287;       % dry-air gas constant, J/(kg*K)
  C.atm.rho0 = 1.225;            % sea-level density, kg/m^3
  C.atm.p0 = 101325.0;           % sea-level pressure, Pa
  C.atm.T0 = 288.15;             % sea-level temperature, K
  C.atm.h_exosphere = 120.0e3;   % drag ignored above this altitude, m

  % Representative 1U CubeSat geometric and mass properties.  The two-body
  % gravity acceleration is independent of spacecraft mass, so these values
  % are documented here for future drag, attitude, and power models.
  C.cubesat.length = 0.10;       % 1U side length, m
  C.cubesat.nominal_mass = 1.33; % typical 1U mass, kg
  C.cubesat.area = 0.01;         % one face area, m^2
  C.cubesat.Cd = 2.2;            % approximate tumbling CubeSat drag coefficient

  % Numerical tolerance used for near-circular and near-equatorial orbit
  % element singularities.
  C.tol = 1.0e-10;
endfunction
