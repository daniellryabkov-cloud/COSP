function mission = rocket_model(C, params)
%ROCKET_MODEL Define launch site, vehicle stages, payload, and guidance.
%
% The default vehicle is an educational three-stage expendable launcher sized
% to place a 1U CubeSat-class payload into an approximate 500 km LEO.  It is
% not a replica of a certified launch vehicle.
%
% Stage fields:
%   dry_mass_kg       : inert structural/engine mass dropped at staging
%   prop_mass_kg      : usable propellant mass
%   thrust_N          : vacuum-equivalent thrust magnitude
%   isp_s             : specific impulse, s
%   Cd                : drag coefficient during that stage
%   reference_area_m2 : aerodynamic reference area
%
% Mass depletion equation:
%   mdot = -T / (Isp g0)
%
% The guidance law is a gravity-turn pitch schedule in the desired launch
% plane.  It is intentionally explicit so future work can replace it with
% closed-loop guidance, finite-burn targeting, or optimal control.

  if nargin < 2
    params = struct();
  endif

  U = vector_utils();

  mission.name = "Scientific CubeSat surface launch to LEO";
  mission.target_altitude_m = getfield_default(params, "target_altitude_m", 500.0e3);
  mission.deployment_delta_v_mps = getfield_default(params, "deployment_delta_v_mps", 1.0);

  mission.launch.latitude_deg = getfield_default(params, "launch_latitude_deg", 28.5);
  mission.launch.longitude_deg = getfield_default(params, "launch_longitude_deg", 0.0);
  mission.launch.azimuth_deg = getfield_default(params, "launch_azimuth_deg", 90.0);

  lat = U.deg2rad(mission.launch.latitude_deg);
  lon = U.deg2rad(mission.launch.longitude_deg);
  az = U.deg2rad(mission.launch.azimuth_deg);

  up = [cos(lat) * cos(lon); cos(lat) * sin(lon); sin(lat)];
  east = [-sin(lon); cos(lon); 0.0];
  north = [-sin(lat) * cos(lon); -sin(lat) * sin(lon); cos(lat)];

  r0 = C.Re * up;

  % Initial inertial velocity is the launch pad velocity from Earth rotation:
  % v = omega_earth x r.
  omega_vec = [0.0; 0.0; C.omega_earth];
  v0 = cross(omega_vec, r0);

  horizontal_launch = U.unit(sin(az) * east + cos(az) * north);
  mission.guidance.plane_normal = U.unit(cross(r0, horizontal_launch));
  mission.guidance.initial_horizontal = horizontal_launch;

  mission.payload.mass_kg = getfield_default(params, "payload_mass_kg", 1.33);
  mission.payload.Cd = getfield_default(params, "payload_Cd", 2.2);
  mission.payload.reference_area_m2 = getfield_default(params, "payload_area_m2", 0.01);

  if isfield(params, "stages")
    mission.stages = params.stages;
  else
    mission.stages = default_stages();
  endif

  total_stage_mass = 0.0;
  for k = 1:numel(mission.stages)
    total_stage_mass += mission.stages(k).dry_mass_kg + mission.stages(k).prop_mass_kg;
  endfor

  mission.initial_mass_kg = mission.payload.mass_kg + total_stage_mass;
  mission.initial_state = [r0; v0; mission.initial_mass_kg];
  mission.launch.r0_eci_m = r0;
  mission.launch.v0_eci_mps = v0;
endfunction

function stages = default_stages()
%DEFAULT_STAGES Baseline three-stage launcher for the example simulation.
  stages(1).name = "Stage 1 booster";
  stages(1).dry_mass_kg = 3000.0;
  stages(1).prop_mass_kg = 27000.0;
  stages(1).thrust_N = 680.0e3;
  stages(1).isp_s = 295.0;
  stages(1).Cd = 0.45;
  stages(1).reference_area_m2 = 5.0;

  stages(2).name = "Stage 2 sustainer";
  stages(2).dry_mass_kg = 900.0;
  stages(2).prop_mass_kg = 7500.0;
  stages(2).thrust_N = 190.0e3;
  stages(2).isp_s = 335.0;
  stages(2).Cd = 0.35;
  stages(2).reference_area_m2 = 2.0;

  stages(3).name = "Stage 3 orbital insertion";
  stages(3).dry_mass_kg = 220.0;
  stages(3).prop_mass_kg = 2300.0;
  stages(3).thrust_N = 45.0e3;
  stages(3).isp_s = 345.0;
  stages(3).Cd = 0.25;
  stages(3).reference_area_m2 = 0.8;
endfunction

function value = getfield_default(s, name, default_value)
%GETFIELD_DEFAULT Read a structure field or return a default.
  if isfield(s, name)
    value = s.(name);
  else
    value = default_value;
  endif
endfunction
