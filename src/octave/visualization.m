function visualization(history, C, mission, params)
%VISUALIZATION Plot and animate the CubeSat launch mission.
%
% Figures:
%   1. 3D Earth, rocket ascent, post-deployment CubeSat trajectory.
%   2. Altitude, velocity magnitude, and specific orbital energy vs time.
%   3. Ground track latitude/longitude.
%   4. Optional smooth 3D animation with camera following the spacecraft.

  if nargin < 4
    params.export_mp4 = false;
    params.animation_filename = "cubesat_launch_animation.mp4";
    params.animation_fps = 30;
    params.animation_stride = 10;
  endif

  plot_trajectory_3d(history, C, mission);
  plot_time_histories(history);
  plot_ground_track(history);

  if isfield(params, "export_mp4") && params.export_mp4
    animate_launch(history, C, mission, params);
  endif
endfunction

function plot_trajectory_3d(history, C, mission)
%PLOT_TRAJECTORY_3D 3D Earth sphere and launch trajectory.
  r = history.position_eci_m;

  figure("name", "3D CubeSat Launch Trajectory");
  draw_earth(C);
  hold on;

  dep_idx = history.deployment_index;
  if isempty(dep_idx) || isnan(dep_idx)
    dep_idx = rows(r);
  endif

  plot3(r(1:dep_idx, 1), r(1:dep_idx, 2), r(1:dep_idx, 3), ...
        "color", [0.9, 0.25, 0.05], "linewidth", 1.5);
  plot3(r(dep_idx:end, 1), r(dep_idx:end, 2), r(dep_idx:end, 3), ...
        "color", [0.05, 0.55, 1.0], "linewidth", 1.5);
  plot3(r(1, 1), r(1, 2), r(1, 3), "go", "markerfacecolor", "g");
  plot3(r(end, 1), r(end, 2), r(end, 3), "ko", ...
        "markerfacecolor", "y", "markersize", 7);

  axis equal;
  grid on;
  xlabel("ECI x, m");
  ylabel("ECI y, m");
  zlabel("ECI z, m");
  title("Surface Launch to Approximate 500 km LEO");
  view(35, 25);
  hold off;
endfunction

function plot_time_histories(history)
%PLOT_TIME_HISTORIES Engineering plots requested by the specification.
  t_min = history.t_s ./ 60.0;

  figure("name", "Launch and Orbit Time Histories");
  subplot(3, 1, 1);
  plot(t_min, history.altitude_m ./ 1000.0, "b-", "linewidth", 1.2);
  grid on;
  xlabel("Time, min");
  ylabel("Altitude, km");
  title("Altitude vs Time");

  subplot(3, 1, 2);
  plot(t_min, history.speed_mps ./ 1000.0, "m-", "linewidth", 1.2);
  grid on;
  xlabel("Time, min");
  ylabel("Speed, km/s");
  title("Velocity Magnitude vs Time");

  subplot(3, 1, 3);
  plot(t_min, history.energy_j_per_kg ./ 1.0e6, "k-", "linewidth", 1.2);
  grid on;
  xlabel("Time, min");
  ylabel("Specific energy, MJ/kg");
  title("Specific Orbital Energy vs Time");
endfunction

function plot_ground_track(history)
%PLOT_GROUND_TRACK Latitude and longitude track.
  if !isfield(history, "ground_track")
    return;
  endif

  figure("name", "Ground Track");
  [lon_plot, lat_plot] = break_dateline(history.ground_track.longitude_deg, ...
                                        history.ground_track.latitude_deg);
  hold on;
  draw_lat_lon_grid();
  plot(lon_plot, lat_plot, "r-", "linewidth", 1.2);
  plot(history.ground_track.longitude_deg(1), history.ground_track.latitude_deg(1), ...
       "go", "markerfacecolor", "g");
  plot(history.ground_track.longitude_deg(end), history.ground_track.latitude_deg(end), ...
       "ko", "markerfacecolor", "y");
  axis([-180, 180, -90, 90]);
  grid on;
  xlabel("Longitude, deg");
  ylabel("Latitude, deg");
  title("Ground Track, Spherical Rotating Earth");
  hold off;
endfunction

function animate_launch(history, C, mission, params)
%ANIMATE_LAUNCH Camera-follow animation and optional MP4 export.
%
% GNU Octave requires video support for VideoWriter.  If VideoWriter is not
% installed, the animation still plays interactively but MP4 export is skipped.

  can_write_video = exist("VideoWriter", "file") == 2;
  if !can_write_video
    warning("VideoWriter not found. Install/load Octave video support to export MP4.");
  endif

  writer = [];
  if can_write_video
    writer = VideoWriter(params.animation_filename);
    writer.FrameRate = params.animation_fps;
    open(writer);
  endif

  r = history.position_eci_m;
  stride = max(1, params.animation_stride);
  frame_ids = 1:stride:rows(r);
  if frame_ids(end) != rows(r)
    frame_ids = [frame_ids, rows(r)];
  endif

  fig = figure("name", "CubeSat Launch Animation");

  for idx = frame_ids
    clf(fig);
    draw_earth(C);
    hold on;
    plot3(r(1:idx, 1), r(1:idx, 2), r(1:idx, 3), ...
          "color", [0.9, 0.25, 0.05], "linewidth", 1.3);
    plot3(r(idx, 1), r(idx, 2), r(idx, 3), "ko", ...
          "markerfacecolor", "y", "markersize", 7);

    axis equal;
    grid on;
    xlabel("ECI x, m");
    ylabel("ECI y, m");
    zlabel("ECI z, m");
    title(sprintf("CubeSat Launch, t = %.1f s, altitude = %.1f km", ...
                  history.t_s(idx), history.altitude_m(idx) / 1000.0));

    % Camera follows the spacecraft by keeping it near the center of view.
    target = r(idx, :);
    span = C.Re + max(700.0e3, history.altitude_m(idx) + 500.0e3);
    xlim([target(1) - span, target(1) + span]);
    ylim([target(2) - span, target(2) + span]);
    zlim([target(3) - span, target(3) + span]);
    view(35, 25);
    drawnow;

    if can_write_video
      frame = getframe(fig);
      writeVideo(writer, frame);
    endif
    hold off;
  endfor

  if can_write_video
    close(writer);
    printf("Animation saved to %s\n", params.animation_filename);
  endif
endfunction

function draw_earth(C)
%DRAW_EARTH Render a simple Earth sphere.
  [xs, ys, zs] = sphere(80);
  surf(C.Re * xs, C.Re * ys, C.Re * zs, ...
       "facecolor", [0.18, 0.35, 0.70], ...
       "edgecolor", "none");
endfunction

function [lon_out, lat_out] = break_dateline(lon, lat)
%BREAK_DATELINE Insert NaNs where longitude jumps across +/-180 deg.
  lon_out = [];
  lat_out = [];
  for k = 1:numel(lon)
    if k > 1 && abs(lon(k) - lon(k - 1)) > 180.0
      lon_out = [lon_out; NaN];
      lat_out = [lat_out; NaN];
    endif
    lon_out = [lon_out; lon(k)];
    lat_out = [lat_out; lat(k)];
  endfor
endfunction

function draw_lat_lon_grid()
%DRAW_LAT_LON_GRID Minimal map grid without requiring mapping packages.
  for latitude = -60:30:60
    plot([-180, 180], [latitude, latitude], "color", [0.75, 0.75, 0.75]);
  endfor
  for longitude = -150:30:150
    plot([longitude, longitude], [-90, 90], "color", [0.75, 0.75, 0.75]);
  endfor
  plot([-180, 180, 180, -180, -180], [-90, -90, 90, 90, -90], ...
       "color", [0.35, 0.35, 0.35]);
endfunction
