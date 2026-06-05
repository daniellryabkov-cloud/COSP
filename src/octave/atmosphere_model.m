function atm = atmosphere_model(altitude_m, C)
%ATMOSPHERE_MODEL Standard atmosphere density, pressure, and temperature.
%
% Input:
%   altitude_m : geometric altitude above spherical Earth radius, m
%   C          : constants structure
%
% Output fields:
%   rho_kgpm3  : atmospheric density, kg/m^3
%   pressure_pa: static pressure, Pa
%   temp_k     : temperature, K
%
% Equations:
%   For a layer with base altitude hb, base temperature Tb, base pressure Pb,
%   and lapse rate L = dT/dh:
%
%   T = Tb + L (h - hb)
%
%   If L = 0:
%     P = Pb exp[-g0 (h - hb) / (R T)]
%
%   If L != 0:
%     P = Pb (Tb / T)^(g0 / (R L))
%
%   rho = P / (R T)
%
% The layer table follows the 1976 Standard Atmosphere up to 84.852 km.
% Above that, a simple exponential extension is used to keep drag continuous
% until the simulator disables drag at C.atm.h_exosphere.

  h = max(0.0, altitude_m);
  R = C.atm.R_air;
  g0 = C.g0;

  layers = [ ...
       0.0, 288.15, 101325.000, -0.0065; ...
   11000.0, 216.65,  22632.060,  0.0000; ...
   20000.0, 216.65,   5474.889,  0.0010; ...
   32000.0, 228.65,    868.019,  0.0028; ...
   47000.0, 270.65,    110.906,  0.0000; ...
   51000.0, 270.65,     66.939, -0.0028; ...
   71000.0, 214.65,      3.956, -0.0020; ...
   84852.0, 186.946,     0.3734, 0.0000];

  if h >= C.atm.h_exosphere
    atm.rho_kgpm3 = 0.0;
    atm.pressure_pa = 0.0;
    atm.temp_k = 186.946;
    return;
  endif

  layer_idx = rows(layers);
  for k = 1:(rows(layers) - 1)
    if h < layers(k + 1, 1)
      layer_idx = k;
      break;
    endif
  endfor

  hb = layers(layer_idx, 1);
  Tb = layers(layer_idx, 2);
  Pb = layers(layer_idx, 3);
  L = layers(layer_idx, 4);

  if h <= 84852.0
    T = Tb + L * (h - hb);
    if abs(L) < 1.0e-12
      P = Pb * exp(-g0 * (h - hb) / (R * Tb));
    else
      P = Pb * (Tb / T)^(g0 / (R * L));
    endif
    rho = P / (R * T);
  else
    % Smooth exponential continuation above the tabulated atmosphere.
    T = 186.946;
    rho_base = layers(end, 3) / (R * layers(end, 2));
    scale_height_m = 7000.0;
    rho = rho_base * exp(-(h - 84852.0) / scale_height_m);
    P = rho * R * T;
  endif

  atm.rho_kgpm3 = rho;
  atm.pressure_pa = P;
  atm.temp_k = T;
endfunction
