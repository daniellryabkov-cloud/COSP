function [history, mission, C] = main(make_plots, user_params)
%MAIN Scientific CubeSat launch simulator in GNU Octave.
%
% Simulates a three-stage rocket launching a 1U CubeSat-class payload from
% Earth's surface into an approximate 500 km Low Earth Orbit.
%
% Usage:
%   [history, mission, C] = main();              % run, plot, no MP4
%   [history, mission, C] = main(false);         % run without figures
%   params.export_mp4 = true;
%   [history, mission, C] = main(true, params);  % export MP4 animation
%
% Custom examples:
%   params.launch_latitude_deg = 28.5;
%   params.launch_azimuth_deg = 90.0;
%   params.target_altitude_m = 500e3;
%   params.deployment_delta_v_mps = 1.0;
%   [history, mission, C] = main(true, params);

  if nargin < 1
    make_plots = true;
    user_params = struct();
  elseif isstruct(make_plots)
    user_params = make_plots;
    make_plots = true;
  elseif nargin < 2
    user_params = struct();
  endif

  C = constants();
  params = merge_params(default_simulation_params(), user_params);

  mission = rocket_model(C, params);

  history = rk4_propagator([], [], [0.0, params.duration_s], params.dt_s, C, mission);
  history.ground_track = ground_track(history.t_s, history.position_eci_m, C);

  save("-mat", "cubesat_launch_output.mat", "history", "mission", "C", "params");

  print_summary(history, mission, C);

  if make_plots
    visualization(history, C, mission, params);
  endif
endfunction

function params = default_simulation_params()
%DEFAULT_SIMULATION_PARAMS Default ascent and output settings.
  params.target_altitude_m = 500.0e3;       % desired LEO altitude, m
  params.duration_s = 3.0 * 3600.0;         % launch plus coast duration, s
  params.dt_s = 1.0;                        % RK4 step, s
  params.launch_latitude_deg = 28.5;        % launch site latitude, deg
  params.launch_longitude_deg = 0.0;        % launch site longitude, deg
  params.launch_azimuth_deg = 90.0;         % due-east launch, deg from north
  params.payload_mass_kg = 1.33;            % 1U CubeSat nominal mass, kg
  params.deployment_delta_v_mps = 1.0;      % CubeSat separation impulse, m/s
  params.export_mp4 = false;                % set true to write MP4
  params.animation_filename = "cubesat_launch_animation.mp4";
  params.animation_fps = 30;
  params.animation_stride = 10;             % draw every Nth propagated sample
endfunction

function params = merge_params(defaults, overrides)
%MERGE_PARAMS Override default simulation fields with user fields.
  params = defaults;
  names = fieldnames(overrides);
  for k = 1:numel(names)
    params.(names{k}) = overrides.(names{k});
  endfor
endfunction

function print_summary(history, mission, C)
%PRINT_SUMMARY Console summary of ascent and final orbit.
  idx = numel(history.t_s);
  final = orbital_elements(history.position_eci_m(idx, :).', ...
                           history.velocity_eci_mps(idx, :).', C);

  perigee_alt_m = final.a_m * (1.0 - final.e) - C.Re;
  apogee_alt_m = final.a_m * (1.0 + final.e) - C.Re;

  printf("\nScientific CubeSat launch simulation complete\n");
  printf("Duration: %.2f h\n", history.t_s(end) / 3600.0);
  printf("Time step target: %.3f s\n", median(diff(history.t_s)));
  printf("Launch site: lat %.3f deg, lon %.3f deg\n", ...
         mission.launch.latitude_deg, mission.launch.longitude_deg);
  printf("Launch azimuth: %.3f deg\n", mission.launch.azimuth_deg);

  if !isnan(history.deployment_time_s)
    printf("CubeSat deployment time: %.2f s\n", history.deployment_time_s);
  endif

  printf("\nFinal state:\n");
  printf("  altitude: %.3f km\n", history.altitude_m(idx) / 1000.0);
  printf("  speed: %.6f km/s\n", history.speed_mps(idx) / 1000.0);
  printf("  specific energy: %.6f MJ/kg\n", history.energy_j_per_kg(idx) / 1.0e6);
  printf("  angular momentum: %.6e m^2/s\n", history.angular_momentum_m2_per_s(idx));
  printf("\nFinal orbital elements:\n");
  printf("  semi-major axis: %.3f km\n", final.a_m / 1000.0);
  printf("  eccentricity: %.9f\n", final.e);
  printf("  inclination: %.6f deg\n", final.i_deg);
  printf("  RAAN: %.6f deg\n", final.raan_deg);
  printf("  argument of periapsis: %.6f deg\n", final.argp_deg);
  printf("  true anomaly: %.6f deg\n", final.nu_deg);
  printf("  perigee altitude: %.3f km\n", perigee_alt_m / 1000.0);
  printf("  apogee altitude: %.3f km\n", apogee_alt_m / 1000.0);
  printf("  period: %.3f min\n", final.period_s / 60.0);
  printf("\nOutput saved to cubesat_launch_output.mat\n\n");
endfunction
