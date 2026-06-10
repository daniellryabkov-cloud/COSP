"""
launch_simulation.py
====================
Complete rocket-launch + CubeSat orbital-insertion simulation.

Physics
-------
- Newton gravity:        a_g = -μ/r³ · r⃗
- Atmospheric drag:      F_d = -½ρv²C_dA v̂
- Tsiolkovsky thrust:    F = ṁ·Isp·g₀
- Orbital mechanics:     vis-viva, eccentricity vector
- RK4 integrator

Mission profile
---------------
  VERTICAL   →  S1_BURN  →  S1_SEP  →  S2_BURN  →
  COAST  →  CIRC_BURN  →  DEPLOY  →  (CubeSat) SEPARATION  →
  STABILIZATION  →  NOMINAL_MISSION  ↔  CORRECTING  ↔  SAFE_MODE

Visualization style: scientific / engineering (MATLAB / NASA)
  - White background
  - Muted engineering colors
  - No decorative effects
  - Telemetry panels slide in after deployment

Output
------
  mission_scientific.gif   (Pillow)
  mission_scientific.mp4   (FFMpegWriter, requires ffmpeg)
  mission_analysis.png     (static analysis charts)

Usage
-----
  pip install numpy matplotlib pillow
  python launch_simulation.py
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import numpy as np
import os, json, time as _wall
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection

# ──────────────────────────────────────────────────────────────
# ── CONSTANTS ─────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
G    = 6.6743e-11      # m³ kg⁻¹ s⁻²   gravitational constant
ME   = 5.972e24        # kg             Earth mass
RE   = 6_371_000.0     # m              Earth mean radius
MU   = G * ME          # m³ s⁻²        standard gravitational parameter
G0   = 9.80665         # m s⁻²         standard gravity (for Isp)
TALT = 500_000.0       # m              target orbit altitude (500 km)
TR   = RE + TALT       # m              target orbit radius

# ──────────────────────────────────────────────────────────────
# ── ENGINEERING COLOR PALETTE ─────────────────────────────────
# ──────────────────────────────────────────────────────────────
C = {
    "bg":           "white",
    "earth_ocean":  "#1f5a99",
    "earth_land":   "#5f7f4f",
    "earth_edge":   "#0a3060",
    "rocket":       "#888888",
    "cubesat":      "#c62828",
    "traj_rocket":  "#303030",
    "traj_cs":      "#c62828",
    "orbit_target": "#6c757d",
    "text":         "#263238",
    "text_dim":     "#546e7a",
    "panel_bg":     "#f5f5f5",
    "panel_border": "#b0bec5",
    "grid":         "#eceff1",
    "plume":        "#e65100",
    "plume_inner":  "#ff8f00",
    "axis":         "#78909c",
    "highlight":    "#1565c0",
    "warn":         "#b71c1c",
    "ok":           "#2e7d32",
    "nominal":      "#0d47a1",
}

# Mission-phase label colors — muted, consistent
PHASE_C = {
    "VERTICAL":          "#37474f",
    "S1_BURN":           "#37474f",
    "S1_SEP":            "#455a64",
    "S2_BURN":           "#37474f",
    "COAST":             "#455a64",
    "CIRC_BURN":         "#37474f",
    "DEPLOY":            "#1b5e20",
    "DONE":              "#1b5e20",
    "SEPARATION":        "#37474f",
    "STABILIZATION":     "#37474f",
    "NOMINAL_MISSION":   "#0d47a1",
    "CORRECTING":        "#b71c1c",
    "SAFE_MODE":         "#b71c1c",
}

# ──────────────────────────────────────────────────────────────
# ── PHYSICS HELPERS ───────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def gravity(pos: np.ndarray) -> np.ndarray:
    """Point-mass gravitational acceleration [m/s²]."""
    r = np.linalg.norm(pos)
    return -MU / r**3 * pos


def air_density(alt: float) -> float:
    """Exponential atmosphere model [kg/m³]."""
    if alt <= 0:       return 1.225
    if alt > 600_000:  return 0.0
    if alt < 11_000:
        T = 288.15 - 0.0065 * alt
        return 1.225 * (T / 288.15) ** 4.256
    if alt < 86_000:
        return 1.225 * np.exp(-alt / 7_000.0)
    return 1.225 * np.exp(-alt / 8_500.0)


def drag_force(vel: np.ndarray, alt: float,
               area: float, cd: float) -> np.ndarray:
    """Aerodynamic drag force vector [N]."""
    rho = air_density(alt)
    v   = np.linalg.norm(vel)
    if v < 1e-4 or rho < 1e-20:
        return np.zeros(2)
    return -0.5 * rho * v**2 * cd * area * (vel / v)


def v_circ(r: float) -> float:
    """Circular orbital speed at radius r [m/s]."""
    return np.sqrt(MU / r)


def orbital_elements(pos: np.ndarray, vel: np.ndarray) -> dict:
    """Compute Keplerian orbital elements from state vectors."""
    r   = np.linalg.norm(pos)
    v   = np.linalg.norm(vel)
    alt = r - RE
    eps = 0.5 * v**2 - MU / r          # specific mechanical energy
    if eps >= 0:
        return dict(a=1e18, e=1.0, pe=alt, ap=1e12, T=1e12, alt=alt)
    a    = -MU / (2.0 * eps)            # semi-major axis
    evec = ((v**2 / MU - 1.0/r) * pos
            - (np.dot(pos, vel) / MU) * vel)
    e    = min(np.linalg.norm(evec), 0.9999)
    T    = 2.0 * np.pi * np.sqrt(a**3 / MU)
    return dict(a=a, e=e,
                pe=a*(1-e) - RE,
                ap=a*(1+e) - RE,
                T=T, alt=alt)


def rk4_integrate(pos: np.ndarray, vel: np.ndarray,
                  mass: float, dt: float,
                  thrust_vec: np.ndarray,
                  area: float, cd: float):
    """Fourth-order Runge-Kutta integration of equations of motion."""
    def deriv(s):
        p, v = s[:2], s[2:]
        alt  = np.linalg.norm(p) - RE
        acc  = (gravity(p)
                + drag_force(v, alt, area, cd) / mass
                + thrust_vec / mass)
        return np.r_[v, acc]
    s  = np.r_[pos, vel]
    k1 = deriv(s)
    k2 = deriv(s + 0.5*dt*k1)
    k3 = deriv(s + 0.5*dt*k2)
    k4 = deriv(s + dt*k3)
    ns = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return ns[:2].copy(), ns[2:].copy()


# ──────────────────────────────────────────────────────────────
# ── ROCKET SIMULATION ─────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

class Rocket:
    """
    Two-stage launch vehicle simulation.

    Guidance program (pitch schedule):
      0 – 500 m      vertical rise
      500 m – 20 km  linear pitch-over  0° → 45° from vertical
      20 – 80 km     gravity turn (follow velocity vector)
      > 80 km        pure horizontal thrust (orbital insertion)

    Stage-2 MECO criterion: horizontal velocity ≥ 0.97 · v_circ(r)
    Circularisation: prograde ΔV at apoapsis until e < 0.002
    """

    # ── Stage 1 — Kerolox main engine ──────────────────────────
    S1_DRY  = 3_500.0          # kg  structural + engine
    S1_PROP = 28_000.0         # kg  propellant
    S1_THR  = 560_000.0        # N   vacuum thrust
    S1_ISP  = 282.0            # s   specific impulse
    S1_AREA = np.pi * 0.95**2  # m²  frontal area

    # ── Stage 2 — Vacuum upper stage ───────────────────────────
    S2_DRY  = 800.0
    S2_PROP = 6_000.0
    S2_THR  = 95_000.0
    S2_ISP  = 348.0
    S2_AREA = np.pi * 0.85**2

    # ── Payload ─────────────────────────────────────────────────
    PL_MASS = 148.0            # kg  CubeSat + adapter + fairing (120 kg)

    def __init__(self):
        # Initial state: equatorial launch site
        self.pos = np.array([0.0, RE])
        self.vel = np.array([465.1, 0.0])    # Earth surface eastward rotation

        self.s1p  = self.S1_PROP
        self.s2p  = self.S2_PROP
        self.pl   = self.PL_MASS
        self.t    = 0.0
        self.phase = "VERTICAL"

        self._s1_sep      = False
        self._fairing_off = False

        # History arrays (populated each step)
        self.hist = dict(
            t=[], pos=[], vel=[], alt=[], spd=[],
            mass=[], thrust=[], drag=[], phase=[]
        )
        self.events       = []
        self.deploy_pos   = None
        self.deploy_vel   = None
        self.deployed     = False

    # ── Derived state ────────────────────────────────────────────
    def total_mass(self) -> float:
        m2 = self.S2_DRY + self.s2p + self.pl
        if self._s1_sep:
            return m2
        return self.S1_DRY + self.s1p + m2

    def alt(self)  -> float:  return np.linalg.norm(self.pos) - RE
    def spd(self)  -> float:  return np.linalg.norm(self.vel)

    # ── Pitch schedule (guidance) ────────────────────────────────
    def _thrust_direction(self) -> np.ndarray:
        alt   = self.alt()
        r_hat = self.pos / np.linalg.norm(self.pos)        # radial outward
        h_hat = np.array([-r_hat[1], r_hat[0]])            # tangential (east)
        v     = self.vel
        vm    = np.linalg.norm(v)

        if alt < 500.0:
            return r_hat                                   # straight up

        if alt < 20_000.0:
            frac  = ((alt - 500.0) / 19_500.0) ** 0.55   # eased pitch-over
            angle = np.radians(45.0) * frac
            return np.cos(angle)*r_hat + np.sin(angle)*h_hat

        if vm > 50.0:
            return v / vm                                  # gravity turn
        return h_hat

    # ── One integration step ─────────────────────────────────────
    def step(self, dt: float):
        alt  = self.alt()
        spd  = self.spd()
        m    = self.total_mass()
        thr  = 0.0
        mdot = 0.0
        area = self.S1_AREA if not self._s1_sep else self.S2_AREA

        # ── Phase state machine ──────────────────────────────────
        if self.phase == "VERTICAL":
            thr  = self.S1_THR
            mdot = thr / (self.S1_ISP * G0)
            if alt > 500.0:
                self.phase = "S1_BURN"

        elif self.phase == "S1_BURN":
            q     = 0.5 * air_density(alt) * spd**2
            throt = 0.78 if q > 45_000 else 1.0           # Max-Q throttle-down
            thr   = self.S1_THR * throt
            mdot  = thr / (self.S1_ISP * G0)
            if self.s1p <= 0.0:
                self.s1p   = 0.0
                self.phase = "S1_SEP"
                self._log("S1_SEP", f"alt={alt/1000:.1f} km  v={spd:.0f} m/s")

        elif self.phase == "S1_SEP":
            self._s1_sep = True
            self.phase   = "S2_BURN"
            self._log("S2_IGNITION", f"alt={alt/1000:.1f} km")

        elif self.phase == "S2_BURN":
            # Jettison fairing above 100 km
            if not self._fairing_off and alt > 100_000.0:
                self._fairing_off = True
                self.pl -= 120.0
                self._log("FAIRING_SEP", f"alt={alt/1000:.1f} km")
            thr  = self.S2_THR
            mdot = thr / (self.S2_ISP * G0)
            # MECO when horizontal velocity ≥ 97% of circular speed
            r_hat  = self.pos / np.linalg.norm(self.pos)
            h_hat  = np.array([-r_hat[1], r_hat[0]])
            v_horiz = abs(np.dot(self.vel, h_hat))
            vc_now  = v_circ(np.linalg.norm(self.pos))
            if v_horiz >= vc_now * 0.97 or self.s2p <= 0.0:
                self.s2p   = max(self.s2p, 0.0)
                self.phase = "COAST"
                el = orbital_elements(self.pos, self.vel)
                self._log("MECO",
                          f"alt={alt/1000:.1f} km  Ap={el['ap']/1000:.0f} km  "
                          f"v={spd:.0f} m/s")

        elif self.phase == "COAST":
            # Wait for apoapsis (radial velocity → 0)
            r_hat = self.pos / np.linalg.norm(self.pos)
            vr    = np.dot(self.vel, r_hat)
            if vr < 50.0 and alt > TALT * 0.90:
                self.phase = "CIRC_BURN"
                self._log("CIRC_START", f"alt={alt/1000:.1f} km")

        elif self.phase == "CIRC_BURN":
            # Prograde burn to circularise — adjust velocity directly
            vc_tgt = v_circ(np.linalg.norm(self.pos))
            dv     = vc_tgt - spd
            if abs(dv) > 1.0:
                dv_step = float(np.clip(dv, -30.0*dt, 30.0*dt))
                v_hat   = self.vel / spd if spd > 1.0 else np.array([1.0, 0.0])
                self.vel = self.vel + dv_step * v_hat
            else:
                el = orbital_elements(self.pos, self.vel)
                self.phase = "DEPLOY"
                self._log("CIRC_DONE",
                          f"e={el['e']:.5f}  alt={alt/1000:.1f} km")

        elif self.phase == "DEPLOY":
            self.deploy_pos = self.pos.copy()
            self.deploy_vel = self.vel.copy()
            self.deployed   = True
            self.phase      = "DONE"
            self._log("CUBESAT_DEPLOY",
                      f"alt={alt/1000:.1f} km  v={spd:.0f} m/s")

        # ── Propellant consumption ────────────────────────────────
        dm = mdot * dt
        if self.phase in ("VERTICAL", "S1_BURN"):
            self.s1p = max(self.s1p - dm, 0.0)
        elif self.phase == "S2_BURN":
            self.s2p = max(self.s2p - dm, 0.0)

        # ── Thrust vector (zero during coast / circ / deploy) ─────
        if self.phase in ("COAST", "CIRC_BURN", "DEPLOY", "DONE", "S1_SEP"):
            tv = np.zeros(2)
        else:
            tv = thr * self._thrust_direction()

        drag_f = drag_force(self.vel, alt, area, 0.35)
        self.pos, self.vel = rk4_integrate(self.pos, self.vel, m, dt,
                                            tv, area, 0.35)
        self.t += dt

        # ── Record history ────────────────────────────────────────
        self.hist["t"].append(self.t)
        self.hist["pos"].append(self.pos.copy())
        self.hist["vel"].append(self.vel.copy())
        self.hist["alt"].append(self.alt() / 1000.0)
        self.hist["spd"].append(self.spd() / 1000.0)
        self.hist["mass"].append(m)
        self.hist["thrust"].append(float(thr))
        self.hist["drag"].append(float(np.linalg.norm(drag_f)))
        self.hist["phase"].append(self.phase)

    def _log(self, name: str, detail: str = ""):
        msg = f"[T+{self.t:7.1f}s]  {name:20s}  {detail}"
        print(" ", msg)
        self.events.append({"t": self.t, "name": name, "detail": detail})

    def run(self) -> bool:
        """Run launch sequence to CubeSat deployment."""
        print("\n" + "="*60)
        print("  LAUNCH SEQUENCE")
        print("="*60)
        last_print = -60.0
        while not self.deployed and self.t < 4_200.0:
            dt = 8.0 if self.phase == "COAST" else (
                 0.25 if self.alt() < 30_000.0 else 0.5)
            self.step(dt)
            if self.t - last_print >= 60.0:
                last_print = self.t
                el = orbital_elements(self.pos, self.vel)
                print(f"  T+{self.t:5.0f}s  {self.phase:<14s}  "
                      f"Alt: {self.alt()/1000:7.1f} km  "
                      f"v: {self.spd()/1000:.3f} km/s  "
                      f"Ap: {el['ap']/1000:.0f} km")
        if self.deployed:
            el = orbital_elements(self.deploy_pos, self.deploy_vel)
            print(f"\n  ✓ CubeSat deployed  T+{self.t:.0f}s  "
                  f"alt={np.linalg.norm(self.deploy_pos)-RE:.0f} m  "
                  f"ecc={el['e']:.5f}")
            return True
        print("\n  ✗ Deployment failed")
        return False


# ──────────────────────────────────────────────────────────────
# ── CUBESAT SIMULATION ────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

class CubeSat:
    """
    6U CubeSat (10 × 20 × 30 cm, ~8 kg).

    Subsystems modelled:
      Power   — solar panels, Li-ion battery, load management
      GNC     — cold-gas thrusters for orbit correction
      Payload — imaging camera (frame counter)

    Operational phases:
      SEPARATION  → STABILIZATION  → NOMINAL_MISSION
                                    ↕ CORRECTING
                                    ↕ SAFE_MODE
    """

    DRY_MASS   = 8.0      # kg
    PROP_INIT  = 0.5      # kg  cold-gas N₂
    AREA       = 0.02     # m²  frontal area
    CD         = 2.2      # –   drag coefficient (tumbling body)
    THR        = 0.5      # N   thruster force
    ISP        = 60.0     # s
    MDOT       = 0.5 / (60.0 * G0)   # kg/s mass flow rate

    BAT_CAP    = 20.0     # Wh
    SOL_AREA   = 0.40     # m²
    SOL_EFF    = 0.295    # –
    IRR        = 1361.0   # W/m²  solar irradiance (1 AU)
    LOAD_BASE  = 1.2      # W   OBC + ADCS + thermal
    LOAD_TX    = 2.5      # W   UHF transmitter (when on)
    LOAD_CAM   = 1.5      # W   imaging camera

    # Correction thresholds
    ALT_TOL    = 4_000.0  # m   altitude deviation → trigger correction
    ECC_TOL    = 0.010    # –   eccentricity tolerance

    def __init__(self, pos: np.ndarray, vel: np.ndarray, t0: float):
        self.pos   = pos.copy()
        self.vel   = vel.copy()
        self.t     = t0
        self.prop  = self.PROP_INIT
        self.bat   = 0.85 * self.BAT_CAP   # 85% initial charge
        self.phase = "SEPARATION"
        self._phase_start = t0

        # Subsystem flags
        self._panels_deployed = False
        self._tx_on           = False
        self._cam_on          = False

        # Orbit correction state
        self._corr_dv    = 0.0
        self._corr_dir   = np.zeros(2)
        self.total_dv    = 0.0
        self.frames      = 0

        # History
        self.hist = dict(
            t=[], pos=[], alt=[], spd=[],
            soc=[], ecc=[], phase=[], solar=[]
        )
        self.events = []

    # ── State ────────────────────────────────────────────────────
    def mass(self)  -> float:  return self.DRY_MASS + self.prop
    def alt(self)   -> float:  return np.linalg.norm(self.pos) - RE
    def spd(self)   -> float:  return np.linalg.norm(self.vel)
    def soc(self)   -> float:  return self.bat / self.BAT_CAP

    # ── Eclipse detection (cylindrical shadow model) ─────────────
    def _in_eclipse(self) -> bool:
        sun = np.array([1.0, 0.0])              # Sun fixed along +X
        proj = np.dot(self.pos, sun)
        if proj > 0:
            return False
        perp2 = np.dot(self.pos, self.pos) - proj**2
        return perp2 < RE**2

    # ── Electrical power subsystem ───────────────────────────────
    def _update_power(self, dt: float):
        eclipse = self._in_eclipse()
        solar   = 0.0
        if not eclipse and self._panels_deployed:
            ang   = np.arctan2(self.pos[1], self.pos[0])
            solar = self.IRR * self.SOL_AREA * self.SOL_EFF * max(0.0, np.cos(ang))
        load = self.LOAD_BASE
        if self._tx_on:  load += self.LOAD_TX
        if self._cam_on: load += self.LOAD_CAM
        self.bat = float(
            np.clip(self.bat + (solar - load) * dt / 3600.0,
                    0.0, self.BAT_CAP)
        )
        return self.soc(), solar, load, eclipse

    # ── Orbit health check ───────────────────────────────────────
    def _check_orbit(self) -> tuple:
        el      = orbital_elements(self.pos, self.vel)
        alt_err = abs(el["alt"] - TALT)
        ecc     = el["e"]
        if alt_err > self.ALT_TOL or ecc > self.ECC_TOL:
            vc_ = v_circ(np.linalg.norm(self.pos))
            dv  = min(abs(vc_ - self.spd()) + ecc * vc_ * 0.5, 25.0)
            v_hat = self.vel / self.spd() if self.spd() > 1 else np.array([0.0, 1.0])
            self._corr_dir = v_hat if el["alt"] < TALT else -v_hat
            self._corr_dv  = dv
            return True, f"Δalt={(el['alt']-TALT)/1000:.1f} km  e={ecc:.4f}"
        return False, "nominal"

    # ── Operational phase state machine ──────────────────────────
    def _update_phase(self, dt: float):
        dp = self.t - self._phase_start
        ph = self.phase

        if ph == "SEPARATION":
            if not self._panels_deployed and dp > 5.0:
                self._panels_deployed = True
                self._log("SOLAR_PANELS_DEPLOYED")
            if dp > 60.0:
                self.phase        = "STABILIZATION"
                self._phase_start = self.t
                self._log("→ STABILIZATION")

        elif ph == "STABILIZATION":
            if dp > 120.0:
                need, reason = self._check_orbit()
                if need:
                    self.phase        = "CORRECTING"
                    self._phase_start = self.t
                    self._log(f"→ CORRECTING  ({reason})")
                else:
                    self.phase        = "NOMINAL_MISSION"
                    self._phase_start = self.t
                    self._tx_on       = True
                    self._cam_on      = True
                    self._log("→ NOMINAL_MISSION")

        elif ph == "NOMINAL_MISSION":
            if self.soc() < 0.10:
                self.phase        = "SAFE_MODE"
                self._phase_start = self.t
                self._cam_on      = False
                self._tx_on       = False
                self._log("→ SAFE_MODE  (low battery)")
            elif int(dp) % 600 < dt:
                need, reason = self._check_orbit()
                if need:
                    self.phase        = "CORRECTING"
                    self._phase_start = self.t
                    self._cam_on      = False
                    self._log(f"→ CORRECTING  ({reason})")

        elif ph == "CORRECTING":
            if self._corr_dv <= 0.0 or self.prop <= 0.0:
                self.phase        = "STABILIZATION"
                self._phase_start = self.t - 110.0
                self._log("→ STABILIZATION  (correction done)")

        elif ph == "SAFE_MODE":
            if self.soc() > 0.45:
                self.phase        = "STABILIZATION"
                self._phase_start = self.t - 110.0
                self._tx_on       = True
                self._log("→ STABILIZATION  (battery recovered)")

    # ── One integration step ─────────────────────────────────────
    def step(self, dt: float) -> dict:
        self.t += dt
        self._update_phase(dt)
        soc, solar, load, eclipse = self._update_power(dt)

        # Correction thrust
        extra_a = np.zeros(2)
        if self.phase == "CORRECTING" and self._corr_dv > 0.0 and self.prop > 0.0:
            a_thr   = self.THR / self.mass() * self._corr_dir
            extra_a = a_thr
            dv_step = np.linalg.norm(a_thr) * dt
            self._corr_dv  = max(self._corr_dv - dv_step, 0.0)
            self.total_dv += dv_step
            self.prop      = max(self.prop - self.MDOT * dt, 0.0)

        # Image capture (one frame every 30 s when sunlit and imaging)
        if (self.phase == "NOMINAL_MISSION"
                and not eclipse and self._cam_on
                and int(self.t) % 30 == 0):
            self.frames += 1

        self.pos, self.vel = rk4_integrate(
            self.pos, self.vel, self.mass(), dt,
            extra_a * self.mass(), self.AREA, self.CD
        )

        el = orbital_elements(self.pos, self.vel)
        self.hist["t"].append(self.t)
        self.hist["pos"].append(self.pos.copy())
        self.hist["alt"].append(self.alt() / 1000.0)
        self.hist["spd"].append(self.spd() / 1000.0)
        self.hist["soc"].append(soc * 100.0)
        self.hist["ecc"].append(el["e"])
        self.hist["phase"].append(self.phase)
        self.hist["solar"].append(solar)

        return dict(alt=self.alt()/1000, spd=self.spd()/1000,
                    soc=soc, solar=solar, eclipse=eclipse,
                    phase=self.phase, ecc=el["e"])

    def _log(self, msg: str):
        print(f"  [CubeSat T+{self.t:.0f}s]  {msg}")
        self.events.append({"t": self.t, "msg": msg})

    def run(self, duration: float = 5700.0, dt: float = 2.0):
        """Run on-orbit operations for `duration` seconds."""
        print("\n" + "="*60)
        print("  ON-ORBIT OPERATIONS")
        print("="*60)
        t_end = self.t + duration
        last  = self.t - 120.0
        while self.t < t_end:
            s = self.step(dt)
            if self.t - last >= 120.0:
                last = self.t
                shad = "ECLIPSE" if s["eclipse"] else "SUNLIT "
                print(f"  T+{self.t:6.0f}s  {s['phase']:<18s}  "
                      f"Alt: {s['alt']:7.2f} km  "
                      f"v: {s['spd']:.4f} km/s  "
                      f"SOC: {s['soc']*100:5.1f}%  "
                      f"Ecc: {s['ecc']:.5f}  {shad}")


# ──────────────────────────────────────────────────────────────
# ── SCIENTIFIC VISUALIZATION ──────────────────────────────────
# ──────────────────────────────────────────────────────────────

# ── Camera zoom schedule ──────────────────────────────────────
# Keyframes: (sim_time_s, half_extent_m)
ZOOM_KF = [
    (0,      16_000),
    (80,    220_000),
    (400,   900_000),
    (900, 13_000_000),
    (1600,15_500_000),
]

def cam_extent(t: float) -> float:
    """Interpolate camera half-extent at simulation time t."""
    for i in range(len(ZOOM_KF) - 1):
        t0, e0 = ZOOM_KF[i]
        t1, e1 = ZOOM_KF[i + 1]
        if t0 <= t < t1:
            alpha = (t - t0) / (t1 - t0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)   # smooth-step
            return e0 + alpha * (e1 - e0)
    return ZOOM_KF[-1][1]


def _make_earth_patches():
    """
    Return a list of (patch, zorder) for Earth rendering.
    Earth ocean + simplified continent polygons.
    Static — drawn once per animation frame on a pre-rendered background.
    """
    patches = []
    # Ocean base
    patches.append((plt.Circle((0, 0), RE, color=C["earth_ocean"],
                                linewidth=0.8, edgecolor=C["earth_edge"], zorder=2), 2))
    # Approximate continent blobs (angle_deg, size_fraction_of_RE)
    continents = [
        (15,  0.13), (70,  0.10), (155, 0.16),
        (225, 0.09), (285, 0.12), (345, 0.10),
    ]
    for ang, frac in continents:
        ar = np.radians(ang)
        rl = RE * 0.993
        patches.append((
            plt.Circle((rl*np.cos(ar), rl*np.sin(ar)),
                       RE * frac,
                       color=C["earth_land"], zorder=3),
            3
        ))
    return patches


def _make_figure():
    """
    Build the figure layout.

    Layout:
      [  main orbital view  ] [ telemetry panel (right) ]
                              [ strip charts            ]

    Returns fig, ax_main, ax_alt, ax_spd, ax_bat, ax_ecc
    """
    fig = plt.figure(figsize=(15, 8.5), facecolor=C["bg"])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Main orbital view: left 62% of figure
    ax_main = fig.add_axes([0.0, 0.0, 0.62, 1.0],
                            facecolor=C["bg"])
    ax_main.set_aspect("equal")
    ax_main.set_facecolor(C["bg"])

    # Right-side strip charts (stacked vertically)
    strip_l = 0.655
    strip_w = 0.325
    strip_h = 0.185
    strip_gap = 0.015
    axes_strip = []
    for row in range(4):
        y0 = 0.05 + (3 - row) * (strip_h + strip_gap)
        ax = fig.add_axes([strip_l, y0, strip_w, strip_h],
                           facecolor=C["bg"])
        axes_strip.append(ax)

    ax_alt, ax_spd, ax_bat, ax_ecc = axes_strip

    # Style all strip axes
    for ax in axes_strip:
        ax.tick_params(axis="both", which="major",
                       labelsize=7, colors=C["text_dim"],
                       length=3, width=0.6)
        ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color(C["panel_border"])
        ax.set_facecolor(C["bg"])
        ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="-")
        ax.set_axisbelow(True)

    return fig, ax_main, ax_alt, ax_spd, ax_bat, ax_ecc


def _style_main_axis(ax):
    """Apply clean scientific style to the main orbital view axes."""
    ax.set_facecolor(C["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    ax.grid(False)


def build_animation(rkt: Rocket,
                    cs:  CubeSat,
                    out_dir: str,
                    fps:     int  = 15,
                    speedup: int  = 15,
                    save_gif: bool = True,
                    save_mp4: bool = True) -> animation.FuncAnimation:
    """
    Build and export the mission animation.

    Parameters
    ----------
    rkt       : completed Rocket simulation
    cs        : completed CubeSat simulation
    out_dir   : output directory
    fps       : frames per second
    speedup   : simulated seconds per rendered frame
    save_gif  : write mission_scientific.gif
    save_mp4  : write mission_scientific.mp4  (requires ffmpeg)
    """
    rh = rkt.hist
    ch = cs.hist

    rt = np.array(rh["t"])
    ct = np.array(ch["t"])
    rp = np.array(rh["pos"])
    cp = np.array(ch["pos"])

    T_total  = max(rt[-1] if len(rt) else 0,
                   ct[-1] if len(ct) else 0)
    n_frames = min(int(T_total / speedup), 1400)
    frame_t  = np.linspace(0, T_total, n_frames)

    print(f"\n[VIZ]  {n_frames} frames  |  {T_total:.0f} s simulation  "
          f"|  ×{speedup} speed-up  |  {fps} fps")

    fig, ax_main, ax_alt, ax_spd, ax_bat, ax_ecc = _make_figure()
    _style_main_axis(ax_main)

    # ── Pre-render static Earth on main axes ──────────────────────
    # (Earth does not move — rendered once and stored as background)
    earth_patches = _make_earth_patches()
    for patch, zo in earth_patches:
        ax_main.add_patch(patch)

    # Target orbit ring (dashed, thin)
    th_ring = np.linspace(0, 2*np.pi, 720)
    ax_main.plot(TR * np.cos(th_ring), TR * np.sin(th_ring),
                 color=C["orbit_target"], linewidth=0.7,
                 linestyle="--", alpha=0.6, zorder=4, label="Target orbit")

    # ── Camera state (smooth interpolation) ───────────────────────
    cam = [0.0, float(RE), 16_000.0]   # [cx, cy, half_extent]

    # ── Dynamic artist containers ─────────────────────────────────
    # These are cleared and redrawn every frame
    dyn = []

    # ── Strip-chart data accumulators (built incrementally) ───────
    # (We pre-compute data; strip charts just slice the arrays.)

    # ── Frame-draw function ───────────────────────────────────────
    def draw_frame(fi: int):
        nonlocal dyn

        # Clear all dynamic artists from previous frame
        for art in dyn:
            try:
                art.remove()
            except Exception:
                pass
        dyn.clear()

        t = frame_t[fi]

        # ── Interpolate rocket state ──────────────────────────────
        ri   = min(int(np.searchsorted(rt, t)), len(rt) - 1)
        rpos = rp[ri]
        rvel = np.array(rh["vel"][ri])
        rph  = rh["phase"][ri]
        rthr = rh["thrust"][ri]
        ralt = rh["alt"][ri]
        rspd = rh["spd"][ri]
        rmss = rh["mass"][ri]

        # ── Interpolate CubeSat state (once deployed) ─────────────
        cs_live = len(ct) > 0 and t >= ct[0]
        ci = 0
        cpos = rpos.copy()
        calt = ralt; cspd = rspd
        csoc = 85.0; cecc = 0.0; cph = ""
        if cs_live:
            ci   = min(int(np.searchsorted(ct, t)), len(cp) - 1)
            cpos = cp[ci]
            calt = ch["alt"][ci]
            cspd = ch["spd"][ci]
            csoc = ch["soc"][ci]
            cecc = ch["ecc"][ci]
            cph  = ch["phase"][ci]

        # ── Camera smooth follow ──────────────────────────────────
        target_obj = cpos if cs_live else rpos
        te = cam_extent(t)
        alpha = 0.07   # EMA smoothing
        cam[0] = cam[0]*(1-alpha) + target_obj[0]*alpha
        cam[1] = cam[1]*(1-alpha) + target_obj[1]*alpha
        cam[2] = cam[2]*(1-alpha) + te*alpha
        cx, cy, ext = cam

        ax_main.set_xlim(cx - ext, cx + ext)
        ax_main.set_ylim(cy - ext, cy + ext)

        # ── Trajectory trail — rocket (thin dark line) ────────────
        trail_n = min(ri + 1, 500)
        if trail_n > 2:
            tx = rp[max(0, ri - trail_n):ri + 1, 0]
            ty = rp[max(0, ri - trail_n):ri + 1, 1]
            ln, = ax_main.plot(tx, ty,
                               color=C["traj_rocket"],
                               linewidth=0.8, alpha=0.7, zorder=5)
            dyn.append(ln)

        # ── Trajectory trail — CubeSat ────────────────────────────
        if cs_live and ci > 2:
            cs_trail = min(ci + 1, 800)
            ctx = cp[max(0, ci - cs_trail):ci + 1, 0]
            cty = cp[max(0, ci - cs_trail):ci + 1, 1]
            ln, = ax_main.plot(ctx, cty,
                               color=C["traj_cs"],
                               linewidth=0.7, alpha=0.6, zorder=5)
            dyn.append(ln)

        # ── Rocket marker ─────────────────────────────────────────
        # Scale marker to ~0.8% of view extent; minimum 1200 m
        mk = max(ext * 0.008, 1200.0)

        if rph not in ("DEPLOY", "DONE", ""):
            # Orientation: align with velocity vector
            vm  = np.linalg.norm(rvel)
            ang = np.arctan2(rvel[1], rvel[0]) if vm > 1 else np.pi/2
            ca, sa = np.cos(ang - np.pi/2), np.sin(ang - np.pi/2)
            R2 = np.array([[ca, -sa], [sa, ca]])

            # Body: narrow rectangle
            bw, bh = mk * 0.25, mk * 1.6
            rect_local = np.array([
                [-bw/2, -bh/2], [bw/2, -bh/2],
                [bw/2,  bh/2], [-bw/2, bh/2]
            ])
            body = (R2 @ rect_local.T).T + rpos
            rkt_body = ax_main.add_patch(
                plt.Polygon(body, fc=C["rocket"], ec="#555555",
                            lw=0.5, zorder=10, alpha=0.92)
            )
            dyn.append(rkt_body)

            # Nose cone: triangle on top
            cone_local = np.array([
                [-bw/2, bh/2], [bw/2, bh/2], [0, bh/2 + bw*1.2]
            ])
            cone = (R2 @ cone_local.T).T + rpos
            dyn.append(ax_main.add_patch(
                plt.Polygon(cone, fc=C["rocket"], ec="#555555",
                            lw=0.5, zorder=10, alpha=0.92)
            ))

            # Engine plume (only when thrusting)
            if rthr > 100:
                pl_len = mk * 1.4
                plume_local = np.array([
                    [-bw * 0.38, -bh/2],
                    [0,          -bh/2 - pl_len],
                    [bw * 0.38,  -bh/2]
                ])
                plume = (R2 @ plume_local.T).T + rpos
                dyn.append(ax_main.add_patch(
                    plt.Polygon(plume, fc=C["plume"],
                                ec="none", alpha=0.75, zorder=9)
                ))
                # Inner bright core
                core_local = plume_local * 0.45
                core = (R2 @ core_local.T).T + rpos
                dyn.append(ax_main.add_patch(
                    plt.Polygon(core, fc=C["plume_inner"],
                                ec="none", alpha=0.65, zorder=9)
                ))

        # ── CubeSat marker ────────────────────────────────────────
        if cs_live:
            csz = max(ext * 0.005, 800.0)
            cx3, cy3 = cpos

            # Body box
            dyn.append(ax_main.add_patch(
                plt.Rectangle(
                    (cx3 - csz/2, cy3 - csz*0.85),
                    csz, csz*1.7,
                    fc=C["cubesat"], ec="#8b0000",
                    lw=0.5, zorder=12, alpha=0.90
                )
            ))

            # Solar panel wings
            panel_color = "#78909c" if csoc < 20 else "#546e7a"
            for side in [-1, 1]:
                dyn.append(ax_main.add_patch(
                    plt.Rectangle(
                        (cx3 + side * csz * 1.0, cy3 - csz * 0.22),
                        csz * 0.85, csz * 0.44,
                        fc=panel_color, ec="#37474f",
                        lw=0.4, zorder=11, alpha=0.88
                    )
                ))

        # ── Phase label (small, muted, upper-left of view) ────────
        ph_now  = cph if cs_live else rph
        ph_col  = PHASE_C.get(ph_now, C["text"])
        txt_x   = cx - ext * 0.92
        txt_top = cy + ext * 0.89

        dyn.append(ax_main.text(
            txt_x, txt_top,
            ph_now.replace("_", " "),
            color=ph_col, fontsize=8,
            fontfamily="monospace",
            fontweight="semibold",
            ha="left", va="top", zorder=30
        ))

        # ── Time label ────────────────────────────────────────────
        dyn.append(ax_main.text(
            txt_x, txt_top - ext * 0.065,
            f"T + {t:,.0f} s",
            color=C["text_dim"], fontsize=7.5,
            fontfamily="monospace",
            ha="left", va="top", zorder=30
        ))

        # ── Quick-read telemetry (top-left of main view) ──────────
        alt_now = calt if cs_live else ralt
        spd_now = cspd if cs_live else rspd
        tel_lines = [
            f"Alt  {alt_now:8.2f} km",
            f"Vel  {spd_now:8.4f} km/s",
        ]
        if cs_live:
            soc_col = (C["ok"] if csoc > 50
                       else C["warn"] if csoc < 20
                       else C["text_dim"])
            tel_lines += [
                f"Bat  {csoc:8.1f} %",
                f"Ecc  {cecc:.6f}",
            ]
        for i, line in enumerate(tel_lines):
            col = (soc_col if "Bat" in line and cs_live else C["text_dim"])
            dyn.append(ax_main.text(
                txt_x,
                txt_top - ext * (0.150 + i * 0.070),
                line,
                color=col, fontsize=7,
                fontfamily="monospace",
                ha="left", va="top", zorder=30
            ))

        # ── Scale bar ─────────────────────────────────────────────
        sb_len = ext * 0.25
        sb_x   = cx - ext * 0.92
        sb_y   = cy - ext * 0.88
        dyn.append(ax_main.plot(
            [sb_x, sb_x + sb_len], [sb_y, sb_y],
            color=C["text_dim"], lw=1.2, solid_capstyle="butt",
            zorder=28
        )[0])
        # Ticks at ends
        tick_h = ext * 0.015
        for xp in [sb_x, sb_x + sb_len]:
            dyn.append(ax_main.plot(
                [xp, xp], [sb_y, sb_y + tick_h],
                color=C["text_dim"], lw=1.0, zorder=28
            )[0])
        dyn.append(ax_main.text(
            sb_x + sb_len/2,
            sb_y + tick_h * 2.0,
            f"{sb_len/1000:.0f} km",
            color=C["text_dim"], fontsize=6.5,
            fontfamily="monospace",
            ha="center", va="bottom", zorder=28
        ))

        # ── Strip charts (right panel) ────────────────────────────
        rm = rt <= t
        cm = (ct <= t) if cs_live else np.zeros(len(ct), dtype=bool)

        # Helper to apply strip-axis style each frame
        def setup_strip(ax, ylabel, title):
            ax.cla()
            ax.set_facecolor(C["bg"])
            ax.tick_params(axis="both", which="major",
                           labelsize=7, colors=C["text_dim"],
                           length=3, width=0.6)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
                sp.set_color(C["panel_border"])
            ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="-")
            ax.set_axisbelow(True)
            ax.set_ylabel(ylabel, color=C["text_dim"], fontsize=7,
                          labelpad=3)
            ax.set_xlabel("Mission time  [min]",
                          color=C["text_dim"], fontsize=6.5, labelpad=2)
            ax.set_title(title, color=C["text"], fontsize=7.5,
                         fontweight="semibold", pad=4, loc="left")

        # Altitude
        setup_strip(ax_alt, "Altitude  [km]", "Altitude vs. time")
        if rm.any():
            ax_alt.plot(rt[rm]/60, np.array(rh["alt"])[rm],
                        color=C["traj_rocket"],
                        lw=0.9, label="Launch vehicle")
        if cs_live and cm.any():
            ax_alt.plot(ct[cm]/60, np.array(ch["alt"])[cm],
                        color=C["traj_cs"],
                        lw=0.9, label="CubeSat")
        ax_alt.axhline(500, color=C["orbit_target"],
                       lw=0.7, ls="--", alpha=0.7, label="500 km target")
        ax_alt.legend(fontsize=5.5, loc="upper left",
                      facecolor=C["bg"], edgecolor=C["panel_border"],
                      handlelength=1.5, labelcolor=C["text_dim"])

        # Velocity
        setup_strip(ax_spd, "Velocity  [km/s]", "Velocity vs. time")
        if rm.any():
            ax_spd.plot(rt[rm]/60, np.array(rh["spd"])[rm],
                        color=C["traj_rocket"], lw=0.9)
        if cs_live and cm.any():
            ax_spd.plot(ct[cm]/60, np.array(ch["spd"])[cm],
                        color=C["traj_cs"], lw=0.9)
        ax_spd.axhline(v_circ(TR)/1000, color=C["orbit_target"],
                       lw=0.7, ls="--", alpha=0.7,
                       label=f"v_c = {v_circ(TR)/1000:.3f} km/s")
        ax_spd.legend(fontsize=5.5, loc="upper left",
                      facecolor=C["bg"], edgecolor=C["panel_border"],
                      handlelength=1.5, labelcolor=C["text_dim"])

        # Battery
        setup_strip(ax_bat, "State of charge  [%]", "Battery state of charge")
        if cs_live and cm.any():
            sv = np.array(ch["soc"])[cm]
            ax_bat.plot(ct[cm]/60, sv,
                        color=C["nominal"], lw=0.9)
            ax_bat.fill_between(ct[cm]/60, sv,
                                color=C["nominal"], alpha=0.08)
        ax_bat.axhline(20, color=C["warn"],
                       lw=0.7, ls="--", alpha=0.8, label="Critical 20%")
        ax_bat.set_ylim(0, 105)
        ax_bat.legend(fontsize=5.5, loc="upper left",
                      facecolor=C["bg"], edgecolor=C["panel_border"],
                      handlelength=1.5, labelcolor=C["text_dim"])

        # Eccentricity
        setup_strip(ax_ecc, "Eccentricity  [–]", "Orbital eccentricity")
        if cs_live and cm.any():
            ax_ecc.plot(ct[cm]/60, np.array(ch["ecc"])[cm],
                        color=C["traj_cs"], lw=0.9)
        ax_ecc.axhline(0.010, color=C["warn"],
                       lw=0.7, ls="--", alpha=0.8, label="Correction threshold")
        ax_ecc.axhline(0.002, color=C["ok"],
                       lw=0.7, ls=":", alpha=0.6, label="Nominal (e < 0.002)")
        ax_ecc.legend(fontsize=5.5, loc="upper right",
                      facecolor=C["bg"], edgecolor=C["panel_border"],
                      handlelength=1.5, labelcolor=C["text_dim"])

        return dyn

    # ── Create animation object ───────────────────────────────────
    anim = animation.FuncAnimation(
        fig, draw_frame,
        frames   = n_frames,
        interval = 1000 // fps,
        blit     = False,
        repeat   = False,
    )

    # ── Export GIF ────────────────────────────────────────────────
    if save_gif:
        gif_path = os.path.join(out_dir, "mission_scientific.gif")
        print(f"[VIZ]  Saving GIF → {gif_path}")
        anim.save(gif_path,
                  writer=animation.PillowWriter(fps=fps),
                  dpi=90)
        sz = os.path.getsize(gif_path) / 1e6
        print(f"[VIZ]  GIF saved  ({sz:.1f} MB)")

    # ── Export MP4 ────────────────────────────────────────────────
    if save_mp4:
        mp4_path = os.path.join(out_dir, "mission_scientific.mp4")
        try:
            writer = animation.FFMpegWriter(
                fps=fps, bitrate=2000,
                extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
            )
            print(f"[VIZ]  Saving MP4 → {mp4_path}")
            anim.save(mp4_path, writer=writer, dpi=110)
            sz = os.path.getsize(mp4_path) / 1e6
            print(f"[VIZ]  MP4 saved  ({sz:.1f} MB)")
        except Exception as ex:
            print(f"[VIZ]  MP4 skipped ({ex})")

    plt.close(fig)
    return anim


# ──────────────────────────────────────────────────────────────
# ── STATIC ANALYSIS PLOTS ─────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def save_analysis_plots(rkt: Rocket, cs: CubeSat, out_dir: str):
    """
    Save a comprehensive mission-analysis figure.
    Style: MATLAB-like white background, muted engineering colors.
    """
    rh = rkt.hist; ch = cs.hist
    rt = np.array(rh["t"]); ct = np.array(ch["t"])
    rp = np.array(rh["pos"]); cp = np.array(ch["pos"])

    fig = plt.figure(figsize=(17, 11), facecolor=C["bg"])
    fig.suptitle(
        "Mission Analysis  —  CubeSat LEO Insertion",
        color=C["text"], fontsize=12, fontweight="semibold", y=0.98
    )

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.07, right=0.97, top=0.94, bottom=0.05
    )

    def make_ax(row, col, colspan=1):
        ax = fig.add_subplot(gs[row, col:col+colspan])
        ax.set_facecolor(C["bg"])
        ax.tick_params(axis="both", which="major",
                       labelsize=8, colors=C["text_dim"],
                       length=3, width=0.6)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color(C["panel_border"])
        ax.grid(True, color=C["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(C["text_dim"])
        ax.yaxis.label.set_color(C["text_dim"])
        ax.title.set_color(C["text"])
        return ax

    # ── 1. Altitude ───────────────────────────────────────────────
    ax = make_ax(0, 0)
    ax.plot(rt/60, rh["alt"], color=C["traj_rocket"], lw=1.0, label="LV")
    ax.plot(ct/60, ch["alt"], color=C["traj_cs"],    lw=1.0, label="CubeSat")
    ax.axhline(500, color=C["orbit_target"], lw=0.7, ls="--", alpha=0.7)
    ax.set_xlabel("Time  [min]"); ax.set_ylabel("Altitude  [km]")
    ax.set_title("Altitude vs. time", fontsize=9)
    ax.legend(fontsize=7, facecolor=C["bg"], edgecolor=C["panel_border"],
              labelcolor=C["text_dim"])

    # ── 2. Velocity ───────────────────────────────────────────────
    ax = make_ax(0, 1)
    ax.plot(rt/60, rh["spd"], color=C["traj_rocket"], lw=1.0)
    ax.plot(ct/60, ch["spd"], color=C["traj_cs"],    lw=1.0)
    ax.axhline(v_circ(TR)/1000, color=C["orbit_target"],
               lw=0.7, ls="--", alpha=0.7,
               label=f"v_c={v_circ(TR)/1000:.3f} km/s")
    ax.set_xlabel("Time  [min]"); ax.set_ylabel("Velocity  [km/s]")
    ax.set_title("Velocity vs. time", fontsize=9)
    ax.legend(fontsize=7, facecolor=C["bg"], edgecolor=C["panel_border"],
              labelcolor=C["text_dim"])

    # ── 3. 2-D Trajectory ─────────────────────────────────────────
    ax = make_ax(0, 2)
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), RE/1000,
                             color=C["earth_ocean"], zorder=2))
    th = np.linspace(0, 2*np.pi, 720)
    ax.plot(TR/1000*np.cos(th), TR/1000*np.sin(th),
            color=C["orbit_target"], lw=0.7, ls="--", alpha=0.6)
    if len(rp):
        ax.plot(rp[:,0]/1000, rp[:,1]/1000,
                color=C["traj_rocket"], lw=0.7, alpha=0.8)
    if len(cp):
        ax.plot(cp[:,0]/1000, cp[:,1]/1000,
                color=C["traj_cs"], lw=0.8, alpha=0.8)
    ax.set_xlabel("X  [km]"); ax.set_ylabel("Y  [km]")
    ax.set_title("Trajectory (ECI 2-D)", fontsize=9)

    # ── 4. Thrust & Drag ──────────────────────────────────────────
    ax = make_ax(1, 0)
    ax.plot(rt/60, np.array(rh["thrust"])/1000,
            color="#37474f", lw=1.0, label="Thrust")
    ax.plot(rt/60, np.array(rh["drag"])/1000,
            color="#78909c", lw=1.0, ls="--", label="Drag")
    ax.set_xlabel("Time  [min]"); ax.set_ylabel("Force  [kN]")
    ax.set_title("Thrust and aerodynamic drag", fontsize=9)
    ax.legend(fontsize=7, facecolor=C["bg"], edgecolor=C["panel_border"],
              labelcolor=C["text_dim"])

    # ── 5. Vehicle mass ───────────────────────────────────────────
    ax = make_ax(1, 1)
    ax.plot(rt/60, rh["mass"], color=C["traj_rocket"], lw=1.0)
    ax.set_xlabel("Time  [min]"); ax.set_ylabel("Mass  [kg]")
    ax.set_title("Launch vehicle mass", fontsize=9)

    # ── 6. Thrust acceleration (g) ────────────────────────────────
    ax = make_ax(1, 2)
    mss = np.maximum(np.array(rh["mass"]), 1.0)
    ax.plot(rt/60, np.array(rh["thrust"]) / mss / G0,
            color=C["traj_rocket"], lw=1.0)
    ax.set_xlabel("Time  [min]"); ax.set_ylabel("Acceleration  [g]")
    ax.set_title("Thrust acceleration", fontsize=9)

    # ── 7. Battery SOC ────────────────────────────────────────────
    ax = make_ax(2, 0)
    ax.plot(ct/60, ch["soc"], color=C["nominal"], lw=1.0)
    ax.fill_between(ct/60, ch["soc"], color=C["nominal"], alpha=0.08)
    ax.axhline(20, color=C["warn"], lw=0.7, ls="--", alpha=0.7,
               label="Critical 20%")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Time  [min]"); ax.set_ylabel("SOC  [%]")
    ax.set_title("Battery state of charge", fontsize=9)
    ax.legend(fontsize=7, facecolor=C["bg"], edgecolor=C["panel_border"],
              labelcolor=C["text_dim"])

    # ── 8. Eccentricity ───────────────────────────────────────────
    ax = make_ax(2, 1)
    ax.plot(ct/60, ch["ecc"], color=C["traj_cs"], lw=1.0)
    ax.axhline(0.010, color=C["warn"], lw=0.7, ls="--", alpha=0.7,
               label="Correction threshold")
    ax.axhline(0.002, color=C["ok"],   lw=0.7, ls=":", alpha=0.6,
               label="Nominal (0.002)")
    ax.set_xlabel("Time  [min]"); ax.set_ylabel("Eccentricity  [–]")
    ax.set_title("Orbital eccentricity", fontsize=9)
    ax.legend(fontsize=7, facecolor=C["bg"], edgecolor=C["panel_border"],
              labelcolor=C["text_dim"])

    # ── 9. Altitude deviation from 500 km ─────────────────────────
    ax = make_ax(2, 2)
    dev = np.array(ch["alt"]) - 500.0
    ax.plot(ct/60, dev, color=C["traj_cs"], lw=1.0)
    ax.axhline( 0, color=C["ok"],   lw=0.8, ls="--", alpha=0.6,
               label="Target")
    ax.axhline( 2, color=C["warn"], lw=0.6, ls=":",  alpha=0.5)
    ax.axhline(-2, color=C["warn"], lw=0.6, ls=":",  alpha=0.5)
    ax.set_xlabel("Time  [min]"); ax.set_ylabel("Deviation  [km]")
    ax.set_title("Altitude deviation from 500 km", fontsize=9)
    ax.legend(fontsize=7, facecolor=C["bg"], edgecolor=C["panel_border"],
              labelcolor=C["text_dim"])

    # ── 10. Phase timeline ────────────────────────────────────────
    ax = make_ax(3, 0, 3)

    all_phases = sorted(set(rh["phase"] + ch["phase"]))
    pmap = {p: i for i, p in enumerate(all_phases)}

    # Muted engineering phase colors (7 distinct)
    phase_palette = [
        "#546e7a", "#455a64", "#607d8b", "#37474f",
        "#78909c", "#263238", "#b0bec5", "#90a4ae",
    ]

    for i in range(len(rt) - 1):
        c = phase_palette[pmap.get(rh["phase"][i], 0) % len(phase_palette)]
        ax.fill_between(
            [rt[i]/60, rt[i+1]/60], [0.5, 0.5], [1.0, 1.0],
            color=c, alpha=0.80
        )
    for i in range(len(ct) - 1):
        c = phase_palette[pmap.get(ch["phase"][i], 0) % len(phase_palette)]
        ax.fill_between(
            [ct[i]/60, ct[i+1]/60], [0.0, 0.0], [0.5, 0.5],
            color=c, alpha=0.80
        )

    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["CubeSat", "Launch vehicle"],
                       color=C["text_dim"], fontsize=8)
    ax.set_xlabel("Mission time  [min]", color=C["text_dim"], fontsize=8)
    ax.set_title("Mission phase timeline", fontsize=9)

    legend_patches = [
        mpatches.Patch(
            color=phase_palette[v % len(phase_palette)], label=k
        )
        for k, v in list(pmap.items())[:8]
    ]
    ax.legend(handles=legend_patches, fontsize=5.5,
              facecolor=C["bg"], edgecolor=C["panel_border"],
              ncol=4, loc="upper right", labelcolor=C["text_dim"])

    path = os.path.join(out_dir, "mission_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print(f"[VIZ]  Analysis plot → {path}")


# ──────────────────────────────────────────────────────────────
# ── MAIN ──────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def main():
    t0      = _wall.time()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Launch Simulation  —  CubeSat LEO Insertion")
    print(f"  Target orbit : {TALT/1000:.0f} km")
    print(f"  Circ. speed  : {v_circ(TR)/1000:.4f} km/s")
    print(f"  Orbital period: {2*np.pi*(TR**3/MU)**0.5/60:.1f} min")
    print("=" * 60)

    # ── Phase 1: Launch ───────────────────────────────────────────
    rkt = Rocket()
    if not rkt.run():
        print("\nMISSION ABORT"); return

    # ── Phase 2: On-orbit operations ──────────────────────────────
    cs = CubeSat(rkt.deploy_pos, rkt.deploy_vel, rkt.t)
    cs.run(duration=5700.0, dt=2.0)

    # ── Mission summary ───────────────────────────────────────────
    el = orbital_elements(cs.pos, cs.vel)
    print("\n" + "="*60)
    print("  MISSION SUMMARY")
    print("="*60)
    print(f"  Altitude      : {cs.alt()/1000:.2f} km")
    print(f"  Speed         : {cs.spd()/1000:.4f} km/s")
    print(f"  Eccentricity  : {el['e']:.6f}")
    print(f"  Period        : {el['T']/60:.1f} min")
    print(f"  Battery SOC   : {cs.soc()*100:.1f} %")
    print(f"  Images taken  : {cs.frames}")
    print(f"  Correction ΔV : {cs.total_dv:.2f} m/s")

    # ── Telemetry log ─────────────────────────────────────────────
    log = {
        "rocket_events":  rkt.events,
        "cubesat_events": cs.events,
        "final_state": {
            "altitude_km":  cs.alt() / 1000,
            "speed_kms":    cs.spd() / 1000,
            "eccentricity": el["e"],
            "period_min":   el["T"] / 60,
            "battery_pct":  cs.soc() * 100,
            "frames":       cs.frames,
            "correction_dv_ms": cs.total_dv,
        }
    }
    log_path = os.path.join(out_dir, "telemetry_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n  Telemetry log → {log_path}")

    # ── Static analysis plots ─────────────────────────────────────
    print("\n[VIZ]  Generating analysis plots ...")
    save_analysis_plots(rkt, cs, out_dir)

    # ── Animated visualization ────────────────────────────────────
    print("\n[VIZ]  Generating animation ...")
    build_animation(
        rkt, cs, out_dir,
        fps=15, speedup=15,
        save_gif=True,
        save_mp4=True,
    )

    print(f"\n  Done in {_wall.time()-t0:.1f} s")
    print(f"  Output directory: {out_dir}")
    print("    mission_scientific.gif   — animated mission")
    print("    mission_scientific.mp4   — video (if ffmpeg installed)")
    print("    mission_analysis.png     — analysis charts")
    print("    telemetry_log.json       — telemetry data")
    print("=" * 60)


if __name__ == "__main__":
    main()