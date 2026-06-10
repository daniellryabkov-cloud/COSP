"""
visualization.py - Mission visualisation using matplotlib.

Renders:
  • Earth (with atmosphere glow)
  • Rocket body (scales with zoom)
  • Stage debris
  • CubeSat
  • Trajectory trail
  • HUD panels: altitude, speed, battery, phase, orbital elements
  • Smooth zoom from launch site → orbital scale
  • Ground track
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow, Wedge
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional, Tuple
import os

from physics import R_EARTH, TARGET_ALTITUDE


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    'bg':           '#0a0e1a',
    'earth_core':   '#1a3a5c',
    'earth_land':   '#2d5a27',
    'earth_ocean':  '#1e3d59',
    'atm_inner':    '#1a4a7a',
    'atm_outer':    '#0a1a2a',
    'rocket_body':  '#e0e0e0',
    'rocket_flame': '#ff6600',
    'cubesat':      '#00d4ff',
    'cubesat_panel':'#4fc3f7',
    'trajectory':   '#ff4444',
    'orbit_plan':   '#44ff88',
    'hud_bg':       '#0d1b2a',
    'hud_text':     '#c0d8f0',
    'hud_accent':   '#00d4ff',
    'hud_warn':     '#ff8800',
    'hud_ok':       '#44dd66',
    'grid':         '#1a2a3a',
    'phase_colors': {
        'PRELAUNCH':         '#888888',
        'LIFTOFF':           '#ff4400',
        'VERTICAL_RISE':     '#ff6600',
        'GRAVITY_TURN':      '#ffaa00',
        'STAGE_1_SEPARATION':'#ff0088',
        'STAGE_2_BURN':      '#ff44ff',
        'COAST':             '#4488ff',
        'CIRCULARISATION_BURN':'#44ffff',
        'CUBESAT_SEPARATION':'#ffff00',
        'MISSION_COMPLETE':  '#44ff88',
        'SEPARATION':        '#ffff44',
        'STABILIZATION':     '#ff8844',
        'NOMINAL_MISSION':   '#44ff88',
        'CORRECTING':        '#ff4444',
        'SAFE_MODE':         '#ff0000',
    },
}


# ── Camera zoom controller ─────────────────────────────────────────────────────

class CameraController:
    """
    Smooth animated camera zoom from launch-pad view to orbital scale.

    Zoom phases:
      0 – 60 s     : close-up on launch site (±15 km)
      60 – 300 s   : medium zoom tracking rocket (±200 km)
      300 – 600 s  : wide view tracking trajectory (±800 km)
      600 s +      : full orbital scale (±12 000 km)
    """

    ZOOM_KEYFRAMES = [
        # (time_s,   half_extent_m)
        (0.0,        15_000.0),
        (60.0,       200_000.0),
        (300.0,      900_000.0),
        (600.0,      12_000_000.0),
        (1200.0,     15_000_000.0),
    ]

    def __init__(self):
        self._target_center  = np.array([0.0, R_EARTH])
        self._current_center = np.array([0.0, R_EARTH])
        self._target_extent  = self.ZOOM_KEYFRAMES[0][1]
        self._current_extent = self.ZOOM_KEYFRAMES[0][1]
        self._smooth_alpha   = 0.08    # EMA smoothing factor

    def update(self, time_s: float, tracked_pos: np.ndarray):
        """
        Compute camera window for given simulation time and tracked object position.

        Returns
        -------
        (center_x, center_y, half_extent)  all in metres
        """
        # Determine target extent from keyframes
        for i in range(len(self.ZOOM_KEYFRAMES) - 1):
            t0, e0 = self.ZOOM_KEYFRAMES[i]
            t1, e1 = self.ZOOM_KEYFRAMES[i + 1]
            if t0 <= time_s < t1:
                alpha = (time_s - t0) / (t1 - t0)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)   # smooth-step
                self._target_extent = e0 + alpha * (e1 - e0)
                break
        else:
            self._target_extent = self.ZOOM_KEYFRAMES[-1][1]

        # Target center: track the vehicle
        self._target_center = tracked_pos.copy()

        # Smooth interpolation (EMA)
        a = self._smooth_alpha
        self._current_center = (a * self._target_center
                                 + (1.0 - a) * self._current_center)
        self._current_extent = (a * self._target_extent
                                 + (1.0 - a) * self._current_extent)

        return (self._current_center[0],
                self._current_center[1],
                self._current_extent)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_earth(ax, cx: float, cy: float, extent: float):
    """Draw Earth with atmosphere layers visible at current zoom."""
    from matplotlib.patches import Circle

    # Only draw components that are visible in current view
    earth_r = R_EARTH

    # Atmosphere layers (ionosphere → troposphere)
    atm_layers = [
        (earth_r + 600_000, 0.03, '#050d1a'),
        (earth_r + 200_000, 0.05, '#071220'),
        (earth_r + 80_000,  0.08, '#0a1a30'),
        (earth_r + 30_000,  0.10, '#0d2240'),
        (earth_r + 12_000,  0.15, '#102a50'),
        (earth_r + 5_000,   0.20, '#1a3860'),
        (earth_r + 1_000,   0.25, '#1e4070'),
    ]
    for r, alpha, color in atm_layers:
        c = Circle((0, 0), r, color=color, alpha=alpha, zorder=1)
        ax.add_patch(c)

    # Earth body gradient (using two circles)
    earth_body = Circle((0, 0), earth_r, color=PALETTE['earth_ocean'], zorder=2)
    ax.add_patch(earth_body)

    # Crude continent outlines (just colored regions, not real shapes)
    # At orbital scale these look like landmass patches
    for angle_deg, size_frac in [(30, 0.15), (80, 0.12), (150, 0.18),
                                  (220, 0.10), (280, 0.14), (340, 0.11)]:
        angle_rad = np.radians(angle_deg)
        r_land = earth_r * 0.99
        x_land = r_land * np.cos(angle_rad)
        y_land = r_land * np.sin(angle_rad)
        land = Circle((x_land, y_land),
                       earth_r * size_frac,
                       color=PALETTE['earth_land'],
                       alpha=0.7,
                       zorder=3)
        ax.add_patch(land)

    # Target orbit circle
    theta = np.linspace(0, 2 * np.pi, 360)
    r_target = R_EARTH + TARGET_ALTITUDE
    ax.plot(r_target * np.cos(theta),
            r_target * np.sin(theta),
            color=PALETTE['orbit_plan'],
            alpha=0.3, linewidth=0.8,
            linestyle='--', zorder=4)


def draw_rocket(ax, position: np.ndarray, velocity: np.ndarray,
                phase: str, extent: float, thrust_on: bool):
    """Draw rocket body scaled to current view."""
    # Scale rocket size to ~0.5% of view
    size = extent * 0.008
    size = max(size, 1000.0)   # minimum 1 km visual size

    x, y = position
    # Orientation: aligned with velocity vector
    v_mag = np.linalg.norm(velocity)
    if v_mag > 1.0:
        vx, vy = velocity / v_mag
        angle = np.arctan2(vy, vx)
    else:
        angle = np.pi / 2.0

    # Rocket body (rectangle)
    body_w = size * 0.3
    body_h = size * 1.5

    # Draw using rotated patch
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.transforms as transforms

    t = plt.matplotlib.transforms.Affine2D().rotate(angle - np.pi/2)

    # Body
    body = patches.FancyBboxPatch(
        (-body_w/2, -body_h/2), body_w, body_h,
        boxstyle="round,pad=0.1",
        facecolor=PALETTE['rocket_body'],
        edgecolor='white',
        linewidth=0.5,
        alpha=0.9,
        zorder=10,
        transform=t + ax.transData,
    )
    # Translate to position
    body.set_transform(
        plt.matplotlib.transforms.Affine2D()
        .rotate(angle - np.pi/2)
        .translate(x, y)
        + ax.transData
    )
    ax.add_patch(body)

    # Flame plume
    if thrust_on:
        flame_len = size * 1.2
        flame_pts = np.array([
            [-body_w*0.4, 0],
            [0,           -flame_len],
            [body_w*0.4,   0],
        ])
        # Rotate flame
        cos_a, sin_a = np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        flame_pts = (R @ flame_pts.T).T + position
        flame = plt.Polygon(flame_pts,
                            facecolor='#ff6600',
                            edgecolor='#ffaa00',
                            alpha=0.8,
                            zorder=9)
        ax.add_patch(flame)
        # Inner flame
        inner_pts = flame_pts * 0.5 + position * 0.5
        flame2 = plt.Polygon(inner_pts,
                              facecolor='#ffee00',
                              alpha=0.7,
                              zorder=9)
        ax.add_patch(flame2)


def draw_cubesat(ax, position: np.ndarray, extent: float,
                 attitude: float, imaging: bool, in_shadow: bool):
    """Draw CubeSat with solar panels."""
    size = extent * 0.005
    size = max(size, 800.0)

    x, y = position

    # Body box
    body = patches.Rectangle(
        (x - size/2, y - size*0.8),
        size, size * 1.6,
        facecolor=PALETTE['cubesat'],
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9,
        zorder=12,
        angle=np.degrees(attitude),
        rotation_point='center',
    )
    ax.add_patch(body)

    # Solar panels
    panel_color = '#4fc3f7' if not in_shadow else '#2a3a4a'
    for side in [-1, 1]:
        panel = patches.Rectangle(
            (x + side * size * 1.2, y - size * 0.3),
            size * 1.0, size * 0.5,
            facecolor=panel_color,
            edgecolor='#88ccff',
            linewidth=0.5,
            alpha=0.85,
            zorder=11,
        )
        ax.add_patch(panel)

    # Camera glow when imaging
    if imaging and not in_shadow:
        glow = Circle((x, y), size * 1.5,
                       color='#00ff88', alpha=0.15, zorder=10)
        ax.add_patch(glow)


# ── HUD panel drawing ─────────────────────────────────────────────────────────

def draw_hud(fig, state: dict, cubesat_state: Optional[dict],
             mission_time: float):
    """
    Draw heads-up display panels on figure using text annotations.
    Returns list of text artists to be cleared each frame.
    """
    artists = []

    # ── Top-left: Mission status ──────────────────────────────────────────────
    phase = state.get('phase', '---')
    phase_color = PALETTE['phase_colors'].get(phase, '#ffffff')

    info_lines = [
        f"T+ {mission_time:7.1f} s",
        f"PHASE: {phase}",
        f"Alt:   {state.get('altitude_km', 0.0):8.2f} km",
        f"Speed: {state.get('speed_kms', 0.0):8.4f} km/s",
        f"Mass:  {state.get('mass_kg', 0.0):8.1f} kg",
    ]
    if 'thrust_N' in state:
        info_lines.append(f"Thrust:{state['thrust_N']:8.0f} N")

    y_start = 0.98
    for i, line in enumerate(info_lines):
        color = phase_color if i == 1 else PALETTE['hud_text']
        t = fig.text(0.01, y_start - i * 0.025, line,
                     color=color,
                     fontsize=7.5,
                     fontfamily='monospace',
                     transform=fig.transFigure,
                     zorder=100)
        artists.append(t)

    # ── Top-right: CubeSat telemetry ──────────────────────────────────────────
    if cubesat_state:
        cs = cubesat_state
        pwr = cs.get('power', {})
        elems = cs.get('elements', {})
        cam = cs.get('camera', {})

        soc = pwr.get('battery_soc', 0.0) * 100.0
        soc_color = (PALETTE['hud_ok'] if soc > 50
                     else PALETTE['hud_warn'] if soc > 20
                     else '#ff2200')

        cs_lines = [
            f"── CUBESAT TM ──",
            f"Phase: {cs.get('phase', '---')}",
            f"Alt:   {cs.get('altitude_km', 0.0):7.2f} km",
            f"Speed: {cs.get('speed_kms', 0.0):7.4f} km/s",
            f"Batt:  {soc:6.1f}%",
            f"Solar: {pwr.get('solar_power_W', 0.0):6.2f} W",
            f"Load:  {pwr.get('load_power_W', 0.0):6.2f} W",
            f"SMA:   {elems.get('semi_major_axis', 0.0)/1000:7.1f} km",
            f"Ecc:   {elems.get('eccentricity', 0.0):.5f}",
            f"Pe:    {elems.get('periapsis', 0.0)/1000:7.1f} km",
            f"Ap:    {elems.get('apoapsis', 0.0)/1000:7.1f} km",
            f"Imgs:  {cam.get('total_frames', 0)}",
            f"{'[IN SHADOW]' if pwr.get('in_shadow') else '[SUNLIT]   '}",
        ]

        y_cs = 0.98
        for i, line in enumerate(cs_lines):
            color = (soc_color if 'Batt' in line
                     else '#aaaaff' if 'CUBESAT' in line
                     else PALETTE['hud_text'])
            t = fig.text(0.75, y_cs - i * 0.025, line,
                         color=color,
                         fontsize=7.5,
                         fontfamily='monospace',
                         transform=fig.transFigure,
                         zorder=100)
            artists.append(t)

    return artists


# ── Main Visualiser class ──────────────────────────────────────────────────────

class MissionVisualiser:
    """
    Animated mission visualiser.

    Call run() to start the animation and save outputs.
    """

    def __init__(self,
                 rocket_history: dict,
                 cubesat_history: dict,
                 cubesat_events: list,
                 rocket_events: list,
                 save_gif:  bool = True,
                 save_mp4:  bool = True,
                 fps:       int  = 20,
                 speed_up:  int  = 10):
        """
        Parameters
        ----------
        rocket_history   : dict   from Rocket.history
        cubesat_history  : dict   from CubeSat.history
        cubesat_events   : list   CubeSat events (from TelemetrySystem)
        rocket_events    : list   Rocket events (MissionEvent list)
        save_gif         : bool   Save mission.gif
        save_mp4         : bool   Save mission.mp4
        fps              : int    Animation frames per second
        speed_up         : int    Simulation time per rendered frame
        """
        self.rh    = rocket_history
        self.ch    = cubesat_history
        self.c_evt = cubesat_events
        self.r_evt = rocket_events
        self.fps   = fps
        self.speed_up = speed_up
        self.save_gif = save_gif
        self.save_mp4 = save_mp4

        self._cam = CameraController()
        self._hud_artists = []

        # Combine positions for trajectory display
        self._r_pos  = np.array(rocket_history.get('pos', [np.array([0, R_EARTH])]))
        self._cs_pos = (np.array(cubesat_history.get('positions', []))
                        if cubesat_history.get('positions') else None)

        # Time arrays
        self._r_time  = np.array(rocket_history.get('time', [0.0]))
        self._cs_time = np.array(cubesat_history.get('time', [0.0]))

        # Total frames
        total_sim_time = (max(self._r_time[-1] if len(self._r_time) > 0 else 0,
                              self._cs_time[-1] if len(self._cs_time) > 0 else 0))
        self._total_frames = max(1, int(total_sim_time / speed_up * fps / fps))
        self._total_frames = min(self._total_frames, 2000)   # cap at 2000 frames

        # Reindex to evenly-spaced animation frames
        self._frame_times = np.linspace(
            0, total_sim_time, self._total_frames)

    def _get_rocket_state_at(self, t: float) -> dict:
        """Interpolate rocket state at time t."""
        rt = self._r_time
        if len(rt) == 0:
            return {}
        idx = min(np.searchsorted(rt, t), len(rt) - 1)
        rh  = self.rh
        return {
            'position':    rh['pos'][idx],
            'velocity':    rh['vel'][idx],
            'altitude_km': rh['alt'][idx],
            'speed_kms':   rh['speed'][idx],
            'mass_kg':     rh['mass'][idx],
            'phase':       rh['phase'][idx],
            'thrust_N':    rh['thrust'][idx],
            'drag_N':      rh['drag'][idx],
        }

    def _get_cubesat_state_at(self, t: float) -> Optional[dict]:
        """Interpolate CubeSat state at time t."""
        ct = self._cs_time
        if len(ct) == 0 or self._cs_pos is None or len(self._cs_pos) == 0:
            return None
        if t < ct[0]:
            return None
        idx = min(np.searchsorted(ct, t), len(ct) - 1)
        ch  = self.ch
        return {
            'position':    self._cs_pos[idx],
            'altitude_km': ch['altitude'][idx],
            'speed_kms':   ch['speed'][idx],
            'phase':       ch['phase'][idx],
            'power':       {'battery_soc': ch['soc'][idx] / 100.0,
                            'solar_power_W': 5.0,
                            'load_power_W':  3.0,
                            'in_shadow': False},
            'elements':    {'semi_major_axis': (R_EARTH + ch['altitude'][idx]*1000),
                            'eccentricity': ch['ecc'][idx],
                            'periapsis': 0.0,
                            'apoapsis':  0.0},
            'camera':      {'total_frames': 0},
        }

    def _setup_figure(self):
        """Create figure layout."""
        fig = plt.figure(figsize=(16, 10), facecolor=PALETTE['bg'])
        fig.patch.set_facecolor(PALETTE['bg'])

        # Main orbital view takes left 65%
        ax_main = fig.add_axes([0.0, 0.0, 0.65, 1.0])
        ax_main.set_facecolor(PALETTE['bg'])
        ax_main.set_aspect('equal')
        ax_main.axis('off')

        # Right panel: telemetry plots
        ax_alt   = fig.add_axes([0.67, 0.75, 0.31, 0.20])
        ax_speed = fig.add_axes([0.67, 0.52, 0.31, 0.20])
        ax_batt  = fig.add_axes([0.67, 0.29, 0.31, 0.20])
        ax_ecc   = fig.add_axes([0.67, 0.06, 0.31, 0.20])

        for ax in [ax_alt, ax_speed, ax_batt, ax_ecc]:
            ax.set_facecolor('#0d1520')
            ax.tick_params(colors=PALETTE['hud_text'], labelsize=7)
            ax.spines[:].set_color('#1a2a3a')

        return fig, ax_main, ax_alt, ax_speed, ax_batt, ax_ecc

    def _update_plots(self, t_now: float,
                      ax_alt, ax_speed, ax_batt, ax_ecc):
        """Update the four telemetry strip-charts."""
        for ax in [ax_alt, ax_speed, ax_batt, ax_ecc]:
            ax.cla()
            ax.set_facecolor('#0d1520')
            ax.tick_params(colors=PALETTE['hud_text'], labelsize=7)
            ax.spines[:].set_color('#1a2a3a')

        # Slice data up to current time
        r_mask = self._r_time <= t_now
        rt     = self._r_time[r_mask]
        r_alt  = np.array(self.rh['alt'])[r_mask]
        r_spd  = np.array(self.rh['speed'])[r_mask]

        # Altitude
        ax_alt.plot(rt / 60, r_alt, color='#ff6644', linewidth=1.0, label='Rocket')
        if (self._cs_pos is not None and len(self._cs_time) > 0):
            cs_mask = self._cs_time <= t_now
            ct = self._cs_time[cs_mask]
            if len(ct) > 0:
                cs_alt  = np.array(self.ch['altitude'])[cs_mask]
                cs_spd  = np.array(self.ch['speed'])[cs_mask]
                cs_soc  = np.array(self.ch['soc'])[cs_mask]
                cs_ecc  = np.array(self.ch['ecc'])[cs_mask]

                ax_alt.plot(ct / 60, cs_alt, color=PALETTE['cubesat'],
                            linewidth=1.0, label='CubeSat')
                ax_speed.plot(ct / 60, cs_spd, color=PALETTE['cubesat'], linewidth=1.0)
                ax_batt.plot(ct / 60, cs_soc, color='#44dd66', linewidth=1.0)
                ax_batt.axhline(20, color='#ff4400', linewidth=0.5, linestyle='--')
                ax_ecc.plot(ct / 60, cs_ecc, color='#ff88ff', linewidth=1.0)
                ax_ecc.axhline(0.005, color='#ff4400', linewidth=0.5, linestyle='--')

        ax_alt.axhline(500, color=PALETTE['orbit_plan'],
                       linewidth=0.7, linestyle='--', alpha=0.6)
        ax_alt.set_ylabel('Alt [km]', color=PALETTE['hud_text'], fontsize=7)
        ax_alt.legend(fontsize=6, facecolor='#0d1520',
                      labelcolor=PALETTE['hud_text'], loc='upper left')

        ax_speed.plot(rt / 60, r_spd, color='#ff6644', linewidth=1.0)
        ax_speed.axhline(orbital_v_kms(), color=PALETTE['orbit_plan'],
                         linewidth=0.7, linestyle='--', alpha=0.6)
        ax_speed.set_ylabel('Speed [km/s]', color=PALETTE['hud_text'], fontsize=7)

        ax_batt.set_ylabel('Battery [%]', color=PALETTE['hud_text'], fontsize=7)
        ax_batt.set_ylim(0, 105)

        ax_ecc.set_ylabel('Eccentricity', color=PALETTE['hud_text'], fontsize=7)
        ax_ecc.set_xlabel('Mission time [min]', color=PALETTE['hud_text'], fontsize=7)


def orbital_v_kms():
    from physics import MU_EARTH, TARGET_RADIUS
    return np.sqrt(MU_EARTH / TARGET_RADIUS) / 1000.0


def render_animation(rocket_history: dict,
                     cubesat_history: dict,
                     rocket_events: list,
                     cubesat_events: list,
                     output_dir: str = '.',
                     fps: int = 20,
                     speed_up: int = 15,
                     save_gif: bool = True,
                     save_mp4: bool = True):
    """
    Build and save the mission animation.

    Parameters
    ----------
    rocket_history   : dict from Rocket.history
    cubesat_history  : dict from CubeSat.history
    rocket_events    : list of MissionEvent
    cubesat_events   : list (from TelemetrySystem._events)
    output_dir       : str  directory for output files
    fps              : int  frames per second
    speed_up         : int  simulation seconds per animation frame
    save_gif         : bool save mission.gif
    save_mp4         : bool save mission.mp4
    """

    os.makedirs(output_dir, exist_ok=True)

    vis = MissionVisualiser(
        rocket_history   = rocket_history,
        cubesat_history  = cubesat_history,
        cubesat_events   = cubesat_events,
        rocket_events    = rocket_events,
        save_gif         = save_gif,
        save_mp4         = save_mp4,
        fps              = fps,
        speed_up         = speed_up,
    )

    fig, ax_main, ax_alt, ax_speed, ax_batt, ax_ecc = vis._setup_figure()
    cam = CameraController()

    r_pos_arr  = np.array(rocket_history.get('pos', [np.array([0, R_EARTH])]))
    r_time_arr = np.array(rocket_history.get('time', [0.0]))
    r_vel_arr  = np.array(rocket_history.get('vel', [np.array([465.1, 0.0])]))
    r_phs_arr  = rocket_history.get('phase', ['PRELAUNCH'])
    r_thr_arr  = np.array(rocket_history.get('thrust', [0.0]))
    r_alt_arr  = np.array(rocket_history.get('alt', [0.0]))
    r_spd_arr  = np.array(rocket_history.get('speed', [0.0]))
    r_mss_arr  = np.array(rocket_history.get('mass', [0.0]))
    r_drg_arr  = np.array(rocket_history.get('drag', [0.0]))

    cs_pos_arr = (np.array(cubesat_history.get('positions', []))
                  if cubesat_history.get('positions') else np.array([]))
    cs_time_arr= np.array(cubesat_history.get('time', []))
    cs_alt_arr = np.array(cubesat_history.get('altitude', []))
    cs_spd_arr = np.array(cubesat_history.get('speed', []))
    cs_soc_arr = np.array(cubesat_history.get('soc', []))
    cs_ecc_arr = np.array(cubesat_history.get('ecc', []))
    cs_phs_arr = cubesat_history.get('phase', [])

    total_sim_time = max(
        r_time_arr[-1] if len(r_time_arr) else 0,
        cs_time_arr[-1] if len(cs_time_arr) else 0,
    )
    frame_times = np.linspace(0, total_sim_time,
                               min(int(total_sim_time / speed_up), 1800))

    print(f"[VIS] Rendering {len(frame_times)} frames  "
          f"(sim {total_sim_time:.0f} s, {speed_up}× speed-up, {fps} fps)")

    static_artists = []    # Persisting artists (Earth)
    dynamic_artists = []   # Cleared each frame

    # ── Pre-draw trajectory paths ──────────────────────────────────────────────
    # These are drawn progressively during animation

    # ── Animation function ────────────────────────────────────────────────────
    def animate(frame_idx: int):
        nonlocal dynamic_artists

        # Clear previous dynamic frame
        for art in dynamic_artists:
            try:
                art.remove()
            except Exception:
                pass
        dynamic_artists = []
        ax_main.cla()
        ax_main.set_facecolor(PALETTE['bg'])
        ax_main.axis('off')

        t_now = frame_times[frame_idx] if frame_idx < len(frame_times) else frame_times[-1]

        # ── Current rocket state ───────────────────────────────────────────────
        r_idx = min(int(np.searchsorted(r_time_arr, t_now)), len(r_time_arr) - 1)
        r_pos   = r_pos_arr[r_idx]
        r_vel   = r_vel_arr[r_idx]
        r_phase = r_phs_arr[r_idx]
        r_thr   = r_thr_arr[r_idx]
        r_alt   = r_alt_arr[r_idx]
        r_spd   = r_spd_arr[r_idx]
        r_mss   = r_mss_arr[r_idx]

        # ── Current CubeSat state ──────────────────────────────────────────────
        cs_active = len(cs_pos_arr) > 0 and len(cs_time_arr) > 0 and t_now >= cs_time_arr[0]
        cs_pos    = None
        cs_idx    = 0
        cs_alt    = 0.0
        cs_spd    = 0.0
        cs_soc    = 0.0
        cs_phase  = ''

        if cs_active:
            cs_idx   = min(int(np.searchsorted(cs_time_arr, t_now)), len(cs_pos_arr) - 1)
            cs_pos   = cs_pos_arr[cs_idx]
            cs_alt   = cs_alt_arr[cs_idx] if len(cs_alt_arr) > cs_idx else 0.0
            cs_spd   = cs_spd_arr[cs_idx] if len(cs_spd_arr) > cs_idx else 0.0
            cs_soc   = cs_soc_arr[cs_idx] if len(cs_soc_arr) > cs_idx else 0.0
            cs_phase = cs_phs_arr[cs_idx] if len(cs_phs_arr) > cs_idx else ''

        # ── Camera update ──────────────────────────────────────────────────────
        track_pos = cs_pos if cs_active else r_pos
        cx, cy, ext = cam.update(t_now, track_pos)

        ax_main.set_xlim(cx - ext, cx + ext)
        ax_main.set_ylim(cy - ext, cy + ext)

        # ── Draw Earth ─────────────────────────────────────────────────────────
        draw_earth(ax_main, cx, cy, ext)

        # ── Draw star field ────────────────────────────────────────────────────
        rng = np.random.default_rng(42)
        star_x = rng.uniform(cx - ext, cx + ext, 200)
        star_y = rng.uniform(cy - ext, cy + ext, 200)
        # Only show stars outside atmosphere
        dist = np.sqrt(star_x**2 + star_y**2)
        mask = dist > R_EARTH + 120_000
        stars = ax_main.scatter(star_x[mask], star_y[mask],
                                 s=0.5, c='white', alpha=0.4, zorder=0)
        dynamic_artists.append(stars)

        # ── Rocket trajectory trail ────────────────────────────────────────────
        trail_len = min(r_idx + 1, 300)
        if trail_len > 1:
            tr_x = r_pos_arr[max(0, r_idx - trail_len):r_idx + 1, 0]
            tr_y = r_pos_arr[max(0, r_idx - trail_len):r_idx + 1, 1]
            trail_line, = ax_main.plot(tr_x, tr_y,
                                        color=PALETTE['trajectory'],
                                        linewidth=0.8, alpha=0.5, zorder=5)
            dynamic_artists.append(trail_line)

        # ── CubeSat trajectory trail ───────────────────────────────────────────
        if cs_active and cs_idx > 1:
            cs_trail_len = min(cs_idx + 1, 500)
            cs_tr_x = cs_pos_arr[max(0, cs_idx - cs_trail_len):cs_idx + 1, 0]
            cs_tr_y = cs_pos_arr[max(0, cs_idx - cs_trail_len):cs_idx + 1, 1]
            cs_trail, = ax_main.plot(cs_tr_x, cs_tr_y,
                                      color=PALETTE['cubesat'],
                                      linewidth=1.0, alpha=0.6, zorder=5)
            dynamic_artists.append(cs_trail)

        # ── Draw rocket ────────────────────────────────────────────────────────
        thrust_on = r_thr > 100.0
        if r_phase not in ['CUBESAT_SEPARATION', 'MISSION_COMPLETE']:
            draw_rocket(ax_main, r_pos, r_vel, r_phase, ext, thrust_on)

        # ── Draw CubeSat ───────────────────────────────────────────────────────
        if cs_active and cs_pos is not None:
            r_vec    = cs_pos
            attitude = np.arctan2(r_vec[1], r_vec[0]) + np.pi
            imaging  = cs_phase == 'NOMINAL_MISSION'
            draw_cubesat(ax_main, cs_pos, ext, attitude,
                         imaging=imaging, in_shadow=False)

        # ── Scale indicator ────────────────────────────────────────────────────
        scale_m   = ext * 0.3
        scale_km  = scale_m / 1000.0
        scale_bar = ax_main.plot(
            [cx - ext * 0.85, cx - ext * 0.85 + scale_m],
            [cy - ext * 0.90, cy - ext * 0.90],
            color='white', linewidth=1.5, alpha=0.6, zorder=20)[0]
        scale_lbl = ax_main.text(
            cx - ext * 0.85 + scale_m / 2,
            cy - ext * 0.88,
            f'{scale_km:.0f} km',
            color='white', fontsize=7, ha='center', alpha=0.7, zorder=20)
        dynamic_artists += [scale_bar, scale_lbl]

        # ── Phase label ────────────────────────────────────────────────────────
        phase_now  = cs_phase if cs_active else r_phase
        phase_col  = PALETTE['phase_colors'].get(phase_now, '#ffffff')
        phase_txt  = ax_main.text(
            cx, cy + ext * 0.92,
            phase_now.replace('_', ' '),
            color=phase_col, fontsize=10, ha='center', va='top',
            fontweight='bold',
            fontfamily='monospace',
            alpha=0.9, zorder=25)
        dynamic_artists.append(phase_txt)

        # ── Time label ────────────────────────────────────────────────────────
        time_txt = ax_main.text(
            cx - ext * 0.95, cy + ext * 0.92,
            f'T+ {t_now:.1f} s',
            color='white', fontsize=8, ha='left', va='top',
            fontfamily='monospace', alpha=0.8, zorder=25)
        dynamic_artists.append(time_txt)

        # ── Telemetry strip-charts (right side) ────────────────────────────────
        # Altitude
        ax_alt.cla(); ax_alt.set_facecolor('#0d1520')
        ax_alt.tick_params(colors=PALETTE['hud_text'], labelsize=7)
        ax_alt.spines[:].set_color('#1a2a3a')
        r_mask = r_time_arr <= t_now
        if r_mask.any():
            ax_alt.plot(r_time_arr[r_mask] / 60, r_alt_arr[r_mask],
                        color='#ff6644', linewidth=0.8, label='Rocket')
        if cs_active:
            cs_mask = cs_time_arr <= t_now
            if cs_mask.any():
                ax_alt.plot(cs_time_arr[cs_mask] / 60, cs_alt_arr[cs_mask],
                            color=PALETTE['cubesat'], linewidth=0.8, label='CubeSat')
        ax_alt.axhline(500, color=PALETTE['orbit_plan'], linewidth=0.6, linestyle='--', alpha=0.5)
        ax_alt.set_ylabel('Alt [km]', color=PALETTE['hud_text'], fontsize=7)
        ax_alt.legend(fontsize=6, facecolor='#0d1520', labelcolor=PALETTE['hud_text'], loc='upper left')
        ax_alt.set_title('ALTITUDE', color=PALETTE['hud_accent'], fontsize=7, pad=2)

        # Speed
        ax_speed.cla(); ax_speed.set_facecolor('#0d1520')
        ax_speed.tick_params(colors=PALETTE['hud_text'], labelsize=7)
        ax_speed.spines[:].set_color('#1a2a3a')
        if r_mask.any():
            ax_speed.plot(r_time_arr[r_mask] / 60, r_spd_arr[r_mask],
                          color='#ff6644', linewidth=0.8)
        if cs_active:
            cs_mask = cs_time_arr <= t_now
            if cs_mask.any():
                ax_speed.plot(cs_time_arr[cs_mask] / 60, cs_spd_arr[cs_mask],
                              color=PALETTE['cubesat'], linewidth=0.8)
        v_circ_kms = orbital_v_kms()
        ax_speed.axhline(v_circ_kms, color=PALETTE['orbit_plan'],
                         linewidth=0.6, linestyle='--', alpha=0.5)
        ax_speed.set_ylabel('Speed [km/s]', color=PALETTE['hud_text'], fontsize=7)
        ax_speed.set_title('SPEED', color=PALETTE['hud_accent'], fontsize=7, pad=2)

        # Battery
        ax_batt.cla(); ax_batt.set_facecolor('#0d1520')
        ax_batt.tick_params(colors=PALETTE['hud_text'], labelsize=7)
        ax_batt.spines[:].set_color('#1a2a3a')
        if cs_active:
            cs_mask = cs_time_arr <= t_now
            if cs_mask.any():
                soc_vals = cs_soc_arr[cs_mask]
                batt_colors = ['#44dd66' if s > 50 else
                                '#ff8800' if s > 20 else '#ff2200'
                                for s in soc_vals]
                ax_batt.plot(cs_time_arr[cs_mask] / 60, soc_vals,
                             color='#44dd66', linewidth=0.8)
                ax_batt.fill_between(cs_time_arr[cs_mask] / 60, soc_vals,
                                      alpha=0.2, color='#44dd66')
        ax_batt.axhline(20, color='#ff4400', linewidth=0.6, linestyle='--', alpha=0.7)
        ax_batt.set_ylim(0, 105)
        ax_batt.set_ylabel('Battery [%]', color=PALETTE['hud_text'], fontsize=7)
        ax_batt.set_title('BATTERY SOC', color=PALETTE['hud_accent'], fontsize=7, pad=2)

        # Eccentricity
        ax_ecc.cla(); ax_ecc.set_facecolor('#0d1520')
        ax_ecc.tick_params(colors=PALETTE['hud_text'], labelsize=7)
        ax_ecc.spines[:].set_color('#1a2a3a')
        if cs_active:
            cs_mask = cs_time_arr <= t_now
            if cs_mask.any():
                ax_ecc.plot(cs_time_arr[cs_mask] / 60, cs_ecc_arr[cs_mask],
                            color='#ff88ff', linewidth=0.8)
        ax_ecc.axhline(0.005, color='#ff4400', linewidth=0.6, linestyle='--', alpha=0.7)
        ax_ecc.set_ylabel('Eccentricity', color=PALETTE['hud_text'], fontsize=7)
        ax_ecc.set_xlabel('T [min]', color=PALETTE['hud_text'], fontsize=7)
        ax_ecc.set_title('ECCENTRICITY', color=PALETTE['hud_accent'], fontsize=7, pad=2)

        # ── HUD text overlay ───────────────────────────────────────────────────
        for art in vis._hud_artists:
            try:
                art.remove()
            except Exception:
                pass
        vis._hud_artists = []

        state_snap = {
            'phase':       r_phase,
            'altitude_km': r_alt,
            'speed_kms':   r_spd,
            'mass_kg':     r_mss,
            'thrust_N':    r_thr,
        }
        cs_snap = None
        if cs_active:
            cs_snap = {
                'phase':       cs_phase,
                'altitude_km': cs_alt,
                'speed_kms':   cs_spd,
                'power':       {'battery_soc': cs_soc / 100.0,
                                'solar_power_W': 5.0,
                                'load_power_W': 3.0,
                                'in_shadow': False},
                'elements':    {'semi_major_axis': R_EARTH + cs_alt * 1000,
                                'eccentricity': (cs_ecc_arr[cs_idx]
                                                 if len(cs_ecc_arr) > cs_idx else 0.0),
                                'periapsis': 0.0, 'apoapsis': 0.0},
                'camera':      {'total_frames': 0},
            }

        vis._hud_artists = draw_hud(fig, state_snap, cs_snap, t_now)

        return dynamic_artists

    # ── Build animation ───────────────────────────────────────────────────────
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames     = len(frame_times),
        interval   = 1000 // fps,
        blit       = False,
        repeat     = False,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    if save_gif:
        gif_path = os.path.join(output_dir, 'mission.gif')
        print(f"[VIS] Saving GIF → {gif_path}")
        writer_gif = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer_gif, dpi=80)
        print(f"[VIS] GIF saved  ({os.path.getsize(gif_path)/1e6:.1f} MB)")

    if save_mp4:
        mp4_path = os.path.join(output_dir, 'mission.mp4')
        print(f"[VIS] Saving MP4 → {mp4_path}")
        try:
            writer_mp4 = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                                  extra_args=['-vcodec', 'libx264'])
            anim.save(mp4_path, writer=writer_mp4, dpi=100)
            print(f"[VIS] MP4 saved  ({os.path.getsize(mp4_path)/1e6:.1f} MB)")
        except Exception as e:
            print(f"[VIS] MP4 save failed (ffmpeg missing?): {e}")
            print("[VIS] Skipping MP4 — install ffmpeg for video output.")

    plt.close(fig)
    return anim