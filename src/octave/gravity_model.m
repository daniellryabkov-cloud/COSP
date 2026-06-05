function a = gravity_model(r_eci_m, C)
%GRAVITY_MODEL Newtonian point-mass gravity acceleration.
%
% Inputs:
%   r_eci_m : spacecraft position vector in Earth-centered inertial frame, m
%   C       : constants structure from constants.m
%
% Output:
%   a       : inertial acceleration vector, m/s^2
%
% Equation:
%   a = -mu * r / |r|^3
%
% Physical meaning:
%   Newton's law gives an attractive central acceleration proportional to
%   Earth's gravitational parameter mu and directed opposite the position
%   vector.  The acceleration magnitude is mu / |r|^2.
%
% Expansion point:
%   Atmospheric drag, J2 perturbation, solar radiation pressure, third-body
%   gravity, and thrust can be added here as additional acceleration terms:
%   a_total = a_two_body + a_drag + a_J2 + a_SRP + ...

  r = r_eci_m(:);
  r_norm = sqrt(sum(r.^2));

  if r_norm <= 0.0
    error("gravity is undefined at Earth's center");
  endif

  a = -C.mu .* r ./ (r_norm^3);
endfunction
