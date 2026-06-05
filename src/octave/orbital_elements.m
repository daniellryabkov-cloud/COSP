function elems = orbital_elements(r_eci_m, v_eci_mps, C)
%ORBITAL_ELEMENTS Classical orbital elements from inertial state vectors.
%
% Inputs:
%   r_eci_m    : position vector in Earth-centered inertial frame, m
%   v_eci_mps  : velocity vector in Earth-centered inertial frame, m/s
%   C          : constants structure
%
% Outputs in elems:
%   a_m        : semi-major axis, m
%   e          : eccentricity magnitude, dimensionless
%   i_rad      : inclination, rad
%   raan_rad   : right ascension of ascending node, rad
%   argp_rad   : argument of periapsis, rad
%   nu_rad     : true anomaly, rad
%   period_s   : Keplerian orbital period for bound elliptical orbit, s
%   energy     : specific orbital mechanical energy, J/kg
%   h_vec      : specific angular momentum vector, m^2/s
%
% Aerospace convention:
%   The inertial z-axis is Earth's north pole.  The ascending node is where
%   the spacecraft crosses the equatorial plane moving northward.

  U = vector_utils();
  mu = C.mu;
  tol = C.tol;

  r = r_eci_m(:);
  v = v_eci_mps(:);
  k_hat = [0.0; 0.0; 1.0];

  r_norm = U.norm3(r);       % geocentric distance, m
  v_norm = U.norm3(v);       % inertial speed, m/s

  % Specific angular momentum h = r x v.
  % h is perpendicular to the orbital plane.  Its magnitude is twice the
  % areal velocity and is conserved in pure two-body motion.
  h_vec = cross(r, v);
  h = U.norm3(h_vec);

  if h <= tol
    error("orbital elements are undefined for near-zero angular momentum");
  endif

  % Node vector n = k x h points toward the ascending node.
  n_vec = cross(k_hat, h_vec);
  n = U.norm3(n_vec);

  % Eccentricity vector points from Earth to periapsis.
  % e_vec = (v x h)/mu - r/|r|
  % Its magnitude is the scalar eccentricity.
  e_vec = cross(v, h_vec) ./ mu - r ./ r_norm;
  e = U.norm3(e_vec);

  % Specific orbital energy epsilon = v^2/2 - mu/r.
  % For elliptical orbits epsilon < 0 and a = -mu/(2*epsilon).
  energy = 0.5 * v_norm^2 - mu / r_norm;
  if abs(energy) > tol
    a_m = -mu / (2.0 * energy);
  else
    a_m = Inf;
  endif

  % Inclination is the angle between h and the inertial z-axis.
  i_rad = U.acos_clamped(h_vec(3) / h);

  % RAAN is measured in the equatorial plane from inertial x-axis to n.
  if n > tol
    raan_rad = U.wrap_to_2pi(atan2(n_vec(2), n_vec(1)));
  else
    % Equatorial orbit: ascending node is singular.
    raan_rad = 0.0;
  endif

  % Argument of periapsis is measured in the orbital plane from n to e_vec.
  if n > tol && e > tol
    sin_argp = dot(cross(n_vec, e_vec), h_vec) / (n * e * h);
    cos_argp = dot(n_vec, e_vec) / (n * e);
    argp_rad = U.wrap_to_2pi(atan2(sin_argp, cos_argp));
  else
    % Circular or equatorial orbit: periapsis direction is singular.
    argp_rad = 0.0;
  endif

  % True anomaly is measured from periapsis to the current position.
  if e > tol
    sin_nu = dot(cross(e_vec, r), h_vec) / (e * r_norm * h);
    cos_nu = dot(e_vec, r) / (e * r_norm);
    nu_rad = U.wrap_to_2pi(atan2(sin_nu, cos_nu));
  elseif n > tol
    % Circular inclined orbit: use argument of latitude instead.
    sin_u = dot(cross(n_vec, r), h_vec) / (n * r_norm * h);
    cos_u = dot(n_vec, r) / (n * r_norm);
    nu_rad = U.wrap_to_2pi(atan2(sin_u, cos_u));
  else
    % Circular equatorial orbit: use true longitude.
    nu_rad = U.wrap_to_2pi(atan2(r(2), r(1)));
  endif

  if isfinite(a_m) && a_m > 0.0 && e < 1.0
    period_s = 2.0 * pi * sqrt(a_m^3 / mu);
  else
    period_s = NaN;
  endif

  elems.a_m = a_m;
  elems.e = e;
  elems.i_rad = i_rad;
  elems.raan_rad = raan_rad;
  elems.argp_rad = argp_rad;
  elems.nu_rad = nu_rad;
  elems.i_deg = U.rad2deg(i_rad);
  elems.raan_deg = U.rad2deg(raan_rad);
  elems.argp_deg = U.rad2deg(argp_rad);
  elems.nu_deg = U.rad2deg(nu_rad);
  elems.period_s = period_s;
  elems.energy_j_per_kg = energy;
  elems.h_vec_m2_per_s = h_vec;
  elems.h_m2_per_s = h;
  elems.e_vec = e_vec;
  elems.node_vec = n_vec;
endfunction
