"""
stage.py - Rocket stage model.

Each stage carries its own propellant, engine(s), and structural mass.
The Tsiolkovsky rocket equation drives all propellant consumption.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stage:
    """
    Single rocket stage.

    Physical model
    --------------
    Thrust equation:
        F = ṁ * Isp * g0          (Isp in seconds, g0 = 9.80665 m/s²)

    Mass flow rate:
        ṁ = F / (Isp * g0)

    Tsiolkovsky Δv budget:
        Δv = Isp * g0 * ln(m0 / m1)
    """

    name:            str
    dry_mass:        float          # structural + engine mass [kg]
    propellant_mass: float          # initial propellant load  [kg]
    thrust:          float          # vacuum thrust             [N]
    isp:             float          # vacuum specific impulse   [s]
    cross_section:   float          # frontal area              [m²]

    # Runtime state (not part of constructor signature)
    _propellant_remaining: float = field(init=False)
    _is_ignited:           bool  = field(init=False, default=False)
    _is_exhausted:         bool  = field(init=False, default=False)
    _burn_time_elapsed:    float = field(init=False, default=0.0)

    G0 = 9.80665   # [m/s²]  standard gravity for Isp calculations

    def __post_init__(self):
        self._propellant_remaining = self.propellant_mass

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_mass(self) -> float:
        """Current total mass of the stage [kg]."""
        return self.dry_mass + self._propellant_remaining

    @property
    def propellant_fraction(self) -> float:
        """Fraction of propellant remaining [0, 1]."""
        if self.propellant_mass == 0:
            return 0.0
        return self._propellant_remaining / self.propellant_mass

    @property
    def is_active(self) -> bool:
        return self._is_ignited and not self._is_exhausted

    @property
    def propellant_remaining(self) -> float:
        return self._propellant_remaining

    @property
    def mass_flow_rate(self) -> float:
        """Propellant mass flow rate [kg/s]."""
        return self.thrust / (self.isp * self.G0)

    @property
    def delta_v_remaining(self) -> float:
        """Remaining Δv budget (Tsiolkovsky) [m/s]."""
        if self._propellant_remaining <= 0:
            return 0.0
        total = self.dry_mass + self._propellant_remaining
        return self.isp * self.G0 * np.log(total / self.dry_mass)

    # ── Control ───────────────────────────────────────────────────────────────

    def ignite(self):
        if not self._is_exhausted:
            self._is_ignited = True

    def shutdown(self):
        self._is_ignited = False

    # ── Integration step ──────────────────────────────────────────────────────

    def burn(self, dt: float, throttle: float = 1.0) -> float:
        """
        Consume propellant over a time step and return the thrust force [N].

        Parameters
        ----------
        dt       : float   Integration step [s].
        throttle : float   Engine throttle setting in [0, 1].

        Returns
        -------
        float  Actual thrust produced [N].
        """
        if not self._is_ignited or self._is_exhausted:
            return 0.0

        throttle = float(np.clip(throttle, 0.0, 1.0))

        mdot = self.mass_flow_rate * throttle
        dm   = mdot * dt

        if dm >= self._propellant_remaining:
            # Stage runs dry
            dm = self._propellant_remaining
            self._propellant_remaining = 0.0
            self._is_exhausted = True
            self._is_ignited   = False
            # Return thrust for the fraction of step that had fuel
            actual_dt = dm / (self.mass_flow_rate * throttle) if (self.mass_flow_rate * throttle) > 0 else 0.0
            return self.thrust * throttle * (actual_dt / dt)
        else:
            self._propellant_remaining -= dm
            self._burn_time_elapsed    += dt
            return self.thrust * throttle

    def get_telemetry(self) -> dict:
        return {
            'name':                self.name,
            'dry_mass':            self.dry_mass,
            'propellant_mass':     self.propellant_mass,
            'propellant_remaining':self._propellant_remaining,
            'propellant_fraction': self.propellant_fraction,
            'thrust':              self.thrust,
            'isp':                 self.isp,
            'is_active':           self.is_active,
            'is_exhausted':        self._is_exhausted,
            'delta_v_remaining':   self.delta_v_remaining,
            'mass_flow_rate':      self.mass_flow_rate,
        }


# ── Pre-built stages for the mission ──────────────────────────────────────────

def make_stage_1() -> Stage:
    """
    First stage — kerolox main engine (Falcon 9 / Electron class).

    Masses roughly representative of a small-lift launcher first stage.
    """
    return Stage(
        name            = "Stage-1 (Kerolox Main Engine)",
        dry_mass        = 3_500.0,      # kg  (structure + engine)
        propellant_mass = 28_000.0,     # kg  RP-1 / LOX
        thrust          = 560_000.0,    # N   ~56 tf vacuum
        isp             = 282.0,        # s   (sea-level Isp ≈ 255 s)
        cross_section   = np.pi * (0.95)**2,   # m²  ~1.9 m diameter
    )


def make_stage_2() -> Stage:
    """
    Second stage — upper-stage Merlin-Vacuum-class engine.
    """
    return Stage(
        name            = "Stage-2 (Upper Stage Vacuum Engine)",
        dry_mass        = 800.0,
        propellant_mass = 6_000.0,
        thrust          = 95_000.0,     # N
        isp             = 348.0,        # s  (vacuum)
        cross_section   = np.pi * (0.85)**2,
    )