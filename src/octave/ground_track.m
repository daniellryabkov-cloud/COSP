function gt = ground_track(t_s, r_eci_m, C, theta0_rad)
%GROUND_TRACK Convert ECI positions into geodetic-like latitude/longitude.
%
% Inputs:
%   t_s        : time vector from propagation, s
%   r_eci_m    : N-by-3 inertial position array, m
%   C          : constants structure
%   theta0_rad : optional Greenwich sidereal angle at t = 0, rad
%
% Outputs:
%   gt.latitude_rad, gt.longitude_rad : spherical Earth coordinates, rad
%   gt.latitude_deg, gt.longitude_deg : spherical Earth coordinates, deg
%   gt.altitude_m                     : radius - Re, m
%
% Equation:
%   r_ecef = R_eci_to_ecef(theta) r_eci
%   theta = theta0 + omega_earth * t
%
% A spherical Earth is assumed for latitude and altitude.  This is adequate
% for a simple mission simulator but not for precision geodesy.

  if nargin < 4
    theta0_rad = 0.0;
  endif

  U = vector_utils();
  N = numel(t_s);
  lat = zeros(N, 1);
  lon = zeros(N, 1);
  alt = zeros(N, 1);

  for k = 1:N
    theta = theta0_rad + C.omega_earth * t_s(k);

    % ECI to ECEF rotation.  A fixed inertial point appears to drift west
    % over Earth, so longitude decreases as theta increases.
    c = cos(theta);
    s = sin(theta);
    R_eci_to_ecef = [ c,  s, 0.0;
                     -s,  c, 0.0;
                    0.0, 0.0, 1.0];

    r_ecef = R_eci_to_ecef * r_eci_m(k, :).';
    r_norm = U.norm3(r_ecef);

    lat(k) = asin(r_ecef(3) / r_norm);
    lon(k) = atan2(r_ecef(2), r_ecef(1));
    alt(k) = r_norm - C.Re;
  endfor

  gt.latitude_rad = lat;
  gt.longitude_rad = lon;
  gt.latitude_deg = U.rad2deg(lat);
  gt.longitude_deg = U.rad2deg(lon);
  gt.altitude_m = alt;
endfunction
