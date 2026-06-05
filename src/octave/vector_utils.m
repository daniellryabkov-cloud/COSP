function U = vector_utils()
%VECTOR_UTILS Small vector and angle utilities for orbital mechanics.
%
% The simulator keeps these utilities in one module so the physics files are
% easy to expand.  Every vector is a 3-by-1 column vector unless noted.

  U.norm3 = @norm3_local;
  U.unit = @unit_local;
  U.clamp = @clamp_local;
  U.acos_clamped = @acos_clamped_local;
  U.wrap_to_2pi = @wrap_to_2pi_local;
  U.rad2deg = @rad2deg_local;
  U.deg2rad = @deg2rad_local;
  U.rot1 = @rot1_local;
  U.rot3 = @rot3_local;
endfunction

function n = norm3_local(v)
%NORM3_LOCAL Euclidean length |v| of a 3D vector.
  n = sqrt(sum(v(:).^2));
endfunction

function u = unit_local(v)
%UNIT_LOCAL Unit vector v/|v|.  Raises an error for zero-length vectors.
  n = norm3_local(v);
  if n <= 0
    error("unit vector is undefined for a zero vector");
  endif
  u = v(:) ./ n;
endfunction

function y = clamp_local(x, lo, hi)
%CLAMP_LOCAL Limit x to the closed interval [lo, hi].
  y = min(max(x, lo), hi);
endfunction

function a = acos_clamped_local(x)
%ACOS_CLAMPED_LOCAL acos with input clamped to [-1, 1] for roundoff safety.
  a = acos(clamp_local(x, -1.0, 1.0));
endfunction

function y = wrap_to_2pi_local(x)
%WRAP_TO_2PI_LOCAL Wrap an angle in radians into [0, 2*pi).
  y = mod(x, 2.0*pi);
  if y < 0
    y = y + 2.0*pi;
  endif
endfunction

function d = rad2deg_local(r)
%RAD2DEG_LOCAL Convert radians to degrees.
  d = r .* 180.0 ./ pi;
endfunction

function r = deg2rad_local(d)
%DEG2RAD_LOCAL Convert degrees to radians.
  r = d .* pi ./ 180.0;
endfunction

function R = rot1_local(angle_rad)
%ROT1_LOCAL Passive/active direction-cosine rotation about the x-axis.
%
% This matrix is used in the standard perifocal-to-ECI transform
% Q = R3(RAAN) * R1(inclination) * R3(argument_of_periapsis).
  c = cos(angle_rad);
  s = sin(angle_rad);
  R = [1.0, 0.0, 0.0;
       0.0,   c,  -s;
       0.0,   s,   c];
endfunction

function R = rot3_local(angle_rad)
%ROT3_LOCAL Direction-cosine rotation about the z-axis.
  c = cos(angle_rad);
  s = sin(angle_rad);
  R = [  c,  -s, 0.0;
         s,   c, 0.0;
       0.0, 0.0, 1.0];
endfunction
