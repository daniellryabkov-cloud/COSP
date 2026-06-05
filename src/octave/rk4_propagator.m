function history = rk4_propagator(r0_eci_m, v0_eci_mps, tspan_s, dt_s, C, model)
%RK4_PROPAGATOR Fixed-step fourth-order Runge-Kutta propagation.
%
% Modes:
%   1. Satellite coast mode:
%      history = rk4_propagator(r0, v0, [t0 tf], dt, C)
%
%   2. Rocket ascent mode:
%      history = rk4_propagator([], [], [t0 tf], dt, C, mission)
%
% State equations for satellite coast:
%   y = [r; v]
%   dr/dt = v
%   dv/dt = -mu r / |r|^3 + a_drag
%
% State equations for rocket ascent:
%   y = [r; v; m]
%   dr/dt = v
%   dv/dt = a_gravity + (F_thrust + F_drag) / m
%   dm/dt = -T / (Isp g0)
%
% RK4 equation:
%   y_next = y + dt (k1 + 2 k2 + 2 k3 + k4) / 6
%
% Stage boundaries are handled as integration events.  The step is shortened
% to end exactly at burnout, dry mass is dropped, and the next stage starts.

  if nargin >= 6 && isstruct(model) && isfield(model, "stages")
    history = propagate_rocket(tspan_s, dt_s, C, model);
  else
    history = propagate_coast(r0_eci_m, v0_eci_mps, tspan_s, dt_s, C);
  endif
endfunction

function history = propagate_coast(r0, v0, tspan_s, dt_s, C)
%PROPAGATE_COAST Two-body plus atmospheric drag coast propagation.
  validate_time(tspan_s, dt_s);
  t = make_time_vector(tspan_s, dt_s);
  N = numel(t);
  state = zeros(N, 6);
  state(1, :) = [r0(:).', v0(:).'];

  for k = 1:(N - 1)
    h = t(k + 1) - t(k);
    y = state(k, :).';
    f = @(tt, yy) coast_derivative(tt, yy, C);
    state(k + 1, :) = rk4_step(f, t(k), y, h).';
  endfor

  history = analyze_state_history(t, state(:, 1:3), state(:, 4:6), C);
  history.mass_kg = NaN(size(t));
  history.stage_index = zeros(size(t));
  history.thrust_N = zeros(size(t));
endfunction

function history = propagate_rocket(tspan_s, dt_s, C, mission)
%PROPAGATE_ROCKET RK4 ascent propagation with discrete staging events.
  validate_time(tspan_s, dt_s);

  t = tspan_s(1);
  tf = tspan_s(2);
  y = mission.initial_state(:);

  stage_idx = 1;
  deployed = false;
  deployment_time_s = NaN;
  final_phase = 0; % 0 inactive, 1 raise apogee, 2 coast, 3 circularize
  prop_remaining = mission.stages(1).prop_mass_kg;

  t_log = [];
  y_log = [];
  stage_log = [];
  thrust_log = [];
  q_log = [];
  rho_log = [];
  final_phase_log = [];

  while t <= tf + 1.0e-9
    diag = rocket_diagnostics(t, y, C, mission, stage_idx, prop_remaining > 0.0);
    t_log = [t_log; t];
    y_log = [y_log; y.'];
    stage_log = [stage_log; stage_idx];
    thrust_log = [thrust_log; diag.thrust_N];
    q_log = [q_log; diag.dynamic_pressure_pa];
    rho_log = [rho_log; diag.rho_kgpm3];
    final_phase_log = [final_phase_log; final_phase];

    if t >= tf
      break;
    endif

    if stage_idx == numel(mission.stages) && prop_remaining > 0.0
      if final_phase == 0
        final_phase = 1;
      elseif final_phase == 2 && final_circularization_allowed(y, C, mission)
        final_phase = 3;
      endif
    endif

    thrusting = (stage_idx <= numel(mission.stages)) && (prop_remaining > 0.0);
    if thrusting && stage_idx == numel(mission.stages)
      thrusting = final_stage_ignition_allowed(y, C, mission, final_phase);
    endif
    h = min(dt_s, tf - t);

    if thrusting
      st = mission.stages(stage_idx);
      mdot = st.thrust_N / (st.isp_s * C.g0);
      h = min(h, prop_remaining / mdot);
    endif

    f = @(tt, yy) rocket_derivative(tt, yy, C, mission, stage_idx, thrusting);
    y = rk4_step(f, t, y, h);
    t = t + h;

    if thrusting
      st = mission.stages(stage_idx);
      mdot = st.thrust_N / (st.isp_s * C.g0);
      prop_remaining = max(0.0, prop_remaining - mdot * h);

      if stage_idx == numel(mission.stages)
        if final_phase == 1 && apogee_raise_reached(y, C, mission)
          final_phase = 2;
        elseif final_phase == 3 && target_orbit_reached(y, C, mission)
          prop_remaining = 0.0;
        endif
      endif

      if prop_remaining <= 1.0e-8
        % Stage separation: dry mass is instantaneously discarded.
        y(7) = y(7) - st.dry_mass_kg;
        stage_idx = stage_idx + 1;

        if stage_idx <= numel(mission.stages)
          prop_remaining = mission.stages(stage_idx).prop_mass_kg;
          if stage_idx == numel(mission.stages)
            final_phase = 1;
          endif
        elseif !deployed
          % CubeSat release from the spent final stage.  A small prograde
          % deployment impulse separates the payload from the upper stage.
          tangent_hat = guidance_tangent(y(1:3), mission);
          y(4:6) = y(4:6) + mission.deployment_delta_v_mps * tangent_hat;
          y(7) = mission.payload.mass_kg;
          deployed = true;
          deployment_time_s = t;
          prop_remaining = 0.0;
        endif
      endif
    endif

    if norm(y(1:3)) < C.Re - 1000.0
      warning("vehicle impacted Earth before the requested final time");
      break;
    endif
  endwhile

  history = analyze_state_history(t_log, y_log(:, 1:3), y_log(:, 4:6), C);
  history.mass_kg = y_log(:, 7);
  history.stage_index = stage_log;
  history.thrust_N = thrust_log;
  history.dynamic_pressure_pa = q_log;
  history.atmospheric_density_kgpm3 = rho_log;
  history.final_stage_phase = final_phase_log;
  history.deployment_time_s = deployment_time_s;
  history.deployment_index = find(t_log >= deployment_time_s, 1);
  history.mission = mission;
endfunction

function dydt = coast_derivative(t, y, C)
%COAST_DERIVATIVE Coast dynamics with gravity and atmospheric drag.
  r = y(1:3);
  v = y(4:6);
  m = C.cubesat.nominal_mass;
  payload.Cd = C.cubesat.Cd;
  payload.reference_area_m2 = C.cubesat.area;
  a = gravity_model(r, C) + drag_acceleration(t, r, v, m, payload, C);
  dydt = [v; a];
endfunction

function dydt = rocket_derivative(t, y, C, mission, stage_idx, thrusting)
%ROCKET_DERIVATIVE Rocket equations of motion for one continuous burn arc.
  r = y(1:3);
  v = y(4:6);
  m = y(7);

  a_gravity = gravity_model(r, C);
  a_drag = drag_acceleration(t, r, v, m, current_aero(mission, stage_idx), C);

  thrust_accel = [0.0; 0.0; 0.0];
  mdot = 0.0;

  if thrusting && stage_idx <= numel(mission.stages)
    st = mission.stages(stage_idx);
    thrust_hat = guidance_direction(t, r, v, C, mission);
    thrust_accel = (st.thrust_N / m) * thrust_hat;

    % Rocket equation mass flow: propellant mass decreases at T/(Isp*g0).
    mdot = -st.thrust_N / (st.isp_s * C.g0);
  endif

  dydt = [v; a_gravity + a_drag + thrust_accel; mdot];
endfunction

function diag = rocket_diagnostics(t, y, C, mission, stage_idx, thrusting)
%ROCKET_DIAGNOSTICS Engineering quantities logged during propagation.
  r = y(1:3);
  v = y(4:6);
  m = y(7);
  aero = current_aero(mission, stage_idx);
  alt = norm(r) - C.Re;
  atm = atmosphere_model(alt, C);
  omega_vec = [0.0; 0.0; C.omega_earth];
  v_atm = cross(omega_vec, r);
  v_rel = v - v_atm;
  q = 0.5 * atm.rho_kgpm3 * dot(v_rel, v_rel);

  diag.dynamic_pressure_pa = q;
  diag.rho_kgpm3 = atm.rho_kgpm3;
  diag.drag_N = q * aero.Cd * aero.reference_area_m2;
  diag.thrust_N = 0.0;
  if thrusting && stage_idx <= numel(mission.stages)
    diag.thrust_N = mission.stages(stage_idx).thrust_N;
  endif
  diag.mass_kg = m;
endfunction

function a_drag = drag_acceleration(t, r, v, m, aero, C)
%DRAG_ACCELERATION Aerodynamic drag in an Earth-rotating atmosphere.
%
% Equation:
%   F_drag = -0.5 rho |v_rel|^2 Cd A v_rel_hat
%   a_drag = F_drag / m
%
% v_rel is velocity relative to the atmosphere, not inertial velocity.  The
% atmosphere is assumed to corotate with Earth:
%   v_atm = omega_earth x r

  alt = norm(r) - C.Re;
  atm = atmosphere_model(alt, C);
  omega_vec = [0.0; 0.0; C.omega_earth];
  v_atm = cross(omega_vec, r);
  v_rel = v - v_atm;
  v_rel_norm = norm(v_rel);

  if atm.rho_kgpm3 <= 0.0 || v_rel_norm <= 1.0e-9 || m <= 0.0
    a_drag = [0.0; 0.0; 0.0];
  else
    drag_force = -0.5 * atm.rho_kgpm3 * v_rel_norm^2 * aero.Cd * ...
                 aero.reference_area_m2 * (v_rel / v_rel_norm);
    a_drag = drag_force / m;
  endif
endfunction

function thrust_hat = guidance_direction(t, r, v, C, mission)
%GUIDANCE_DIRECTION Gravity-turn pitch law in the target orbital plane.
%
% The unit vector is decomposed into local radial and in-plane tangential
% directions:
%   thrust_hat = sin(gamma) r_hat + cos(gamma) theta_hat
%
% gamma is the flight-path angle above the local horizontal.  The schedule
% starts nearly vertical, pitches over through the dense atmosphere, and then
% flies nearly horizontal for orbital insertion.

  r_hat = r / norm(r);
  theta_hat = guidance_tangent(r, mission);
  alt = norm(r) - C.Re;
  radial_speed = dot(v, r_hat);

  if alt < 150.0
    gamma_deg = 89.0;
  elseif alt < 20.0e3
    gamma_deg = 89.0 - 44.0 * (alt - 150.0) / 19850.0;
  elseif alt < 80.0e3
    gamma_deg = 45.0 - 35.0 * (alt - 20.0e3) / 60.0e3;
  elseif alt < 150.0e3
    gamma_deg = 10.0 - 10.0 * (alt - 80.0e3) / 70.0e3;
  else
    % Above the sensible atmosphere, damp radial speed and fly nearly
    % horizontal for orbital insertion.
    gamma_deg = max(-8.0, min(4.0, -radial_speed / 220.0));
  endif

  gamma = gamma_deg * pi / 180.0;
  thrust_hat = sin(gamma) * r_hat + cos(gamma) * theta_hat;
  thrust_hat = thrust_hat / norm(thrust_hat);
endfunction

function reached = target_orbit_reached(y, C, mission)
%TARGET_ORBIT_REACHED Upper-stage cutoff condition for approximate LEO.
%
% The shutdown logic compares current inertial speed with circular speed at
% the current radius:
%   v_circ = sqrt(mu / r)
%
% It also requires altitude near the requested target and modest radial
% velocity, preventing cutoff during a steep suborbital climb.
  r = y(1:3);
  v = y(4:6);
  radius = norm(r);
  alt = radius - C.Re;
  r_hat = r / radius;
  radial_speed = dot(v, r_hat);
  speed = norm(v);
  circular_speed = sqrt(C.mu / radius);
  [perigee_alt_m, apogee_alt_m, is_bound] = apsides_from_state(r, v, C);

  reached = is_bound && ...
            (alt >= mission.target_altitude_m - 180.0e3) && ...
            (alt <= mission.target_altitude_m + 250.0e3) && ...
            (perigee_alt_m >= mission.target_altitude_m - 180.0e3) && ...
            (apogee_alt_m <= mission.target_altitude_m + 250.0e3) && ...
            (speed >= 0.970 * circular_speed) && ...
            (speed <= 1.030 * circular_speed) && ...
            (abs(radial_speed) <= 350.0);
endfunction

function [perigee_alt_m, apogee_alt_m, is_bound] = apsides_from_state(r, v, C)
%APSIDES_FROM_STATE Perigee and apogee altitude from osculating elements.
  elems = orbital_elements(r, v, C);
  is_bound = isfinite(elems.a_m) && elems.a_m > 0.0 && elems.e < 1.0;
  if is_bound
    perigee_alt_m = elems.a_m * (1.0 - elems.e) - C.Re;
    apogee_alt_m = elems.a_m * (1.0 + elems.e) - C.Re;
  else
    perigee_alt_m = NaN;
    apogee_alt_m = NaN;
  endif
endfunction

function allowed = final_stage_ignition_allowed(y, C, mission, final_phase)
%FINAL_STAGE_IGNITION_ALLOWED Coast upper stage toward apogee before ignition.
%
% Phase 1 burns after leaving the densest atmosphere to raise apogee.  Phase
% 2 coasts.  Phase 3 burns near apogee to raise perigee and circularize.
  r = y(1:3);
  v = y(4:6);
  radius = norm(r);
  alt = radius - C.Re;
  r_hat = r / radius;
  radial_speed = dot(v, r_hat);

  if final_phase == 1
    allowed = alt >= 120.0e3;
  elseif final_phase == 3
    allowed = (alt >= mission.target_altitude_m - 80.0e3) && ...
              (abs(radial_speed) <= 350.0);
  else
    allowed = false;
  endif
endfunction

function reached = apogee_raise_reached(y, C, mission)
%APOGEE_RAISE_REACHED Stop first upper-stage burn once apogee is near target.
  [perigee_alt_m, apogee_alt_m, is_bound] = apsides_from_state(y(1:3), y(4:6), C);
  reached = is_bound && ...
            (apogee_alt_m >= mission.target_altitude_m + 20.0e3) && ...
            (perigee_alt_m >= 90.0e3);
endfunction

function allowed = final_circularization_allowed(y, C, mission)
%FINAL_CIRCULARIZATION_ALLOWED Restart near target apogee for circularization.
  r = y(1:3);
  v = y(4:6);
  radius = norm(r);
  alt = radius - C.Re;
  r_hat = r / radius;
  radial_speed = dot(v, r_hat);

  allowed = (alt >= mission.target_altitude_m - 60.0e3) && ...
            (radial_speed <= 120.0);
endfunction

function theta_hat = guidance_tangent(r, mission)
%GUIDANCE_TANGENT Prograde horizontal direction in the launch plane.
  r_hat = r / norm(r);
  h_hat = mission.guidance.plane_normal;
  theta_hat = cross(h_hat, r_hat);
  theta_hat = theta_hat / norm(theta_hat);
endfunction

function aero = current_aero(mission, stage_idx)
%CURRENT_AERO Select stage or payload aerodynamic properties.
  if stage_idx <= numel(mission.stages)
    aero.Cd = mission.stages(stage_idx).Cd;
    aero.reference_area_m2 = mission.stages(stage_idx).reference_area_m2;
  else
    aero.Cd = mission.payload.Cd;
    aero.reference_area_m2 = mission.payload.reference_area_m2;
  endif
endfunction

function y_next = rk4_step(f, t, y, h)
%RK4_STEP One fourth-order Runge-Kutta step.
  k1 = f(t, y);
  k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
  k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
  k4 = f(t + h, y + h * k3);
  y_next = y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
endfunction

function history = analyze_state_history(t, r, v, C)
%ANALYZE_STATE_HISTORY Compute outputs required by the mission specification.
  N = numel(t);
  speed = sqrt(sum(v.^2, 2));
  radius = sqrt(sum(r.^2, 2));

  gravity_acceleration = zeros(N, 3);
  altitude_m = radius - C.Re;
  energy_j_per_kg = zeros(N, 1);
  h_vec = zeros(N, 3);
  h_norm = zeros(N, 1);

  elements.a_m = zeros(N, 1);
  elements.e = zeros(N, 1);
  elements.i_rad = zeros(N, 1);
  elements.raan_rad = zeros(N, 1);
  elements.argp_rad = zeros(N, 1);
  elements.nu_rad = zeros(N, 1);
  elements.period_s = zeros(N, 1);

  for k = 1:N
    rk = r(k, :).';
    vk = v(k, :).';
    gravity_acceleration(k, :) = gravity_model(rk, C).';
    energy_j_per_kg(k) = 0.5 * speed(k)^2 - C.mu / radius(k);
    hk = cross(rk, vk);
    h_vec(k, :) = hk.';
    h_norm(k) = norm(hk);

    elems = orbital_elements(rk, vk, C);
    elements.a_m(k) = elems.a_m;
    elements.e(k) = elems.e;
    elements.i_rad(k) = elems.i_rad;
    elements.raan_rad(k) = elems.raan_rad;
    elements.argp_rad(k) = elems.argp_rad;
    elements.nu_rad(k) = elems.nu_rad;
    elements.period_s(k) = elems.period_s;
  endfor

  U = vector_utils();
  elements.i_deg = U.rad2deg(elements.i_rad);
  elements.raan_deg = U.rad2deg(elements.raan_rad);
  elements.argp_deg = U.rad2deg(elements.argp_rad);
  elements.nu_deg = U.rad2deg(elements.nu_rad);

  history.t_s = t;
  history.position_eci_m = r;
  history.velocity_eci_mps = v;
  history.acceleration_eci_mps2 = finite_difference_acceleration(t, v);
  history.gravity_acceleration_eci_mps2 = gravity_acceleration;
  history.radius_m = radius;
  history.speed_mps = speed;
  history.altitude_m = altitude_m;
  history.energy_j_per_kg = energy_j_per_kg;
  history.angular_momentum_vec_m2_per_s = h_vec;
  history.angular_momentum_m2_per_s = h_norm;
  history.orbital_elements = elements;
endfunction

function acceleration = finite_difference_acceleration(t, v)
%FINITE_DIFFERENCE_ACCELERATION Kinematic acceleration dv/dt from velocity.
%
% This captures thrust, drag, gravity, and staging impulses as represented in
% the propagated velocity history.  Central differences are used internally,
% with one-sided differences at the endpoints.
  N = numel(t);
  acceleration = zeros(N, 3);

  if N < 2
    return;
  endif

  acceleration(1, :) = (v(2, :) - v(1, :)) ./ (t(2) - t(1));
  acceleration(N, :) = (v(N, :) - v(N - 1, :)) ./ (t(N) - t(N - 1));

  for k = 2:(N - 1)
    acceleration(k, :) = (v(k + 1, :) - v(k - 1, :)) ./ (t(k + 1) - t(k - 1));
  endfor
endfunction

function validate_time(tspan_s, dt_s)
%VALIDATE_TIME Check time input.
  if dt_s <= 0.0
    error("dt_s must be positive");
  endif
  if numel(tspan_s) != 2 || tspan_s(2) <= tspan_s(1)
    error("tspan_s must be [start, end] with end > start");
  endif
endfunction

function t = make_time_vector(tspan_s, dt_s)
%MAKE_TIME_VECTOR Construct a fixed-step time vector with exact final time.
  t = (tspan_s(1):dt_s:tspan_s(2)).';
  if t(end) < tspan_s(2)
    t = [t; tspan_s(2)];
  endif
endfunction
