"""
power.py - Electrical power subsystem (EPS) for CubeSat.

Models:
  • Solar panels (3U body-mounted, 3 deployable panels)
  • Li-ion battery pack
  • Power distribution to subsystems
  • Eclipse / sunlit detection (simplified 2-D geometry)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict

# Solar constant at 1 AU
SOLAR_IRRADIANCE = 1361.0   # W/m²


@dataclass
class SolarPanel:
    """
    Single solar panel face.

    Parameters
    ----------
    area         : float  Panel area [m²]
    efficiency   : float  Photoelectric conversion efficiency [0–1]
    normal_angle : float  Panel normal direction in body frame [rad]
                         0 = +X,  π/2 = +Y
    """
    area:         float
    efficiency:   float
    normal_angle: float   # body-frame normal, radians

    def power_output(self,
                     sun_angle_rad: float,
                     altitude_m:   float,
                     shadow:       bool) -> float:
        """
        Compute instantaneous power output.

        Parameters
        ----------
        sun_angle_rad : float  Angle of Sun direction in inertial frame [rad].
        altitude_m    : float  Orbital altitude (for 1/r² irradiance correction).
        shadow        : bool   True when spacecraft is in Earth's shadow.

        Returns
        -------
        float  Power [W].
        """
        if shadow:
            return 0.0

        # Cosine of incidence angle between panel normal and Sun direction
        # Panel normal in inertial frame = body frame angle + attitude offset
        # (we assume nadir-pointing, so body +Z ≈ nadir, panels face prograde/anti-normal)
        incidence_cos = np.cos(sun_angle_rad - self.normal_angle)
        incidence_cos = max(0.0, incidence_cos)

        # Irradiance (negligible altitude correction at LEO)
        irr = SOLAR_IRRADIANCE   # W/m²  (LEO ≈ 1361 W/m²)

        return irr * self.area * self.efficiency * incidence_cos


@dataclass
class Battery:
    """
    Li-ion battery pack.

    Parameters
    ----------
    capacity_Wh     : float  Total energy capacity [Wh]
    initial_charge  : float  State of charge at t=0  [0–1]
    max_charge_rate : float  Maximum charge current power [W]
    """
    capacity_Wh:     float
    initial_charge:  float = 1.0
    max_charge_rate: float = 10.0

    _soc: float = field(init=False)   # state of charge [0, 1]

    def __post_init__(self):
        self._soc = float(np.clip(self.initial_charge, 0.0, 1.0))

    @property
    def state_of_charge(self) -> float:
        return self._soc

    @property
    def energy_stored_Wh(self) -> float:
        return self._soc * self.capacity_Wh

    def update(self, net_power_W: float, dt_s: float):
        """
        Update battery SoC given net power flow.

        Parameters
        ----------
        net_power_W : float  Positive = charging, negative = discharging [W].
        dt_s        : float  Time step [s].
        """
        # Clamp charge rate
        if net_power_W > self.max_charge_rate:
            net_power_W = self.max_charge_rate

        delta_Wh = net_power_W * (dt_s / 3600.0)
        new_energy = self.energy_stored_Wh + delta_Wh
        new_soc    = new_energy / self.capacity_Wh
        self._soc  = float(np.clip(new_soc, 0.0, 1.0))

    def is_critically_low(self) -> bool:
        return self._soc < 0.10

    def is_full(self) -> bool:
        return self._soc >= 0.99


class PowerSystem:
    """
    Full EPS: solar array + battery + load management.

    Power budget (nominal):
    ──────────────────────
    • OBC (on-board computer)  : 0.8 W
    • UHF radio (TX)           : 2.5 W
    • UHF radio (RX standby)   : 0.1 W
    • Camera (imaging)         : 1.5 W
    • ADCS (attitude control)  : 0.5 W
    • Propulsion (thruster)    : 3.0 W  (when active)
    • Thermal                  : 0.3 W
    • Margin                   : 0.3 W
    """

    SUBSYSTEM_POWER: Dict[str, float] = {
        'obc':       0.8,
        'radio_rx':  0.1,
        'radio_tx':  2.5,
        'camera':    1.5,
        'adcs':      0.5,
        'propulsion':3.0,
        'thermal':   0.3,
        'margin':    0.3,
    }

    def __init__(self):
        # Body-mounted panels (6U CubeSat, 3 deployable wings)
        self.panels = [
            SolarPanel(area=0.060, efficiency=0.295, normal_angle=0.0),        # +X face
            SolarPanel(area=0.060, efficiency=0.295, normal_angle=np.pi),      # −X face
            SolarPanel(area=0.100, efficiency=0.295, normal_angle=np.pi/2),    # deployable +Y
            SolarPanel(area=0.100, efficiency=0.295, normal_angle=-np.pi/2),   # deployable −Y
            SolarPanel(area=0.040, efficiency=0.295, normal_angle=np.pi/4),    # body +Z/45°
        ]

        self.battery = Battery(
            capacity_Wh    = 20.0,    # ~20 Wh (two 18650-class cells)
            initial_charge = 0.85,
            max_charge_rate= 12.0,
        )

        self._active_subsystems  = {'obc', 'radio_rx', 'adcs', 'thermal', 'margin'}
        self._solar_power_W      = 0.0
        self._load_power_W       = 0.0
        self._net_power_W        = 0.0
        self._in_shadow          = False

        self._history_soc        = []
        self._history_solar      = []
        self._history_load       = []
        self._history_time       = []

    # ── Subsystem control ─────────────────────────────────────────────────────

    def enable_subsystem(self, name: str):
        if name in self.SUBSYSTEM_POWER:
            self._active_subsystems.add(name)

    def disable_subsystem(self, name: str):
        self._active_subsystems.discard(name)

    def is_subsystem_active(self, name: str) -> bool:
        return name in self._active_subsystems

    # ── Shadow / eclipse ─────────────────────────────────────────────────────

    @staticmethod
    def in_shadow(position: np.ndarray, sun_direction: np.ndarray) -> bool:
        """
        Geometric cylindrical shadow model.

        Returns True if the spacecraft is within Earth's umbra cylinder
        (simplified: uses cone approximation).

        Parameters
        ----------
        position      : np.ndarray [x, y] in metres
        sun_direction : np.ndarray unit vector pointing from Earth to Sun
        """
        from physics import R_EARTH
        # Project position onto Sun-direction axis
        proj = np.dot(position, sun_direction)
        if proj > 0:
            return False   # On sunlit side of Earth
        # Perpendicular distance from shadow axis
        perp2 = np.dot(position, position) - proj**2
        return perp2 < R_EARTH**2

    # ── Update step ───────────────────────────────────────────────────────────

    def update(self, dt: float, mission_time: float,
               position: np.ndarray,
               sun_angle: float) -> dict:
        """
        Advance EPS by one time step.

        Parameters
        ----------
        dt           : float  Time step [s].
        mission_time : float  Elapsed mission time [s].
        position     : np.ndarray  Spacecraft position [m].
        sun_angle    : float  Sun direction angle in inertial frame [rad].

        Returns
        -------
        dict  Snapshot of power system state.
        """
        sun_direction = np.array([np.cos(sun_angle), np.sin(sun_angle)])
        self._in_shadow = self.in_shadow(position, sun_direction)

        # Solar array output
        solar_total = 0.0
        for panel in self.panels:
            solar_total += panel.power_output(sun_angle, 0.0, self._in_shadow)
        self._solar_power_W = solar_total

        # Load
        load_total = 0.0
        for sub in self._active_subsystems:
            load_total += self.SUBSYSTEM_POWER.get(sub, 0.0)
        self._load_power_W = load_total

        # Net power
        self._net_power_W = self._solar_power_W - self._load_power_W
        self.battery.update(self._net_power_W, dt)

        # History
        self._history_time.append(mission_time)
        self._history_soc.append(self.battery.state_of_charge)
        self._history_solar.append(self._solar_power_W)
        self._history_load.append(self._load_power_W)

        return self.get_state()

    # ── Telemetry snapshot ────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            'battery_soc':    self.battery.state_of_charge,
            'battery_Wh':     self.battery.energy_stored_Wh,
            'solar_power_W':  self._solar_power_W,
            'load_power_W':   self._load_power_W,
            'net_power_W':    self._net_power_W,
            'in_shadow':      self._in_shadow,
            'panels_active':  len(self.panels),
            'subsystems':     list(self._active_subsystems),
            'battery_critical': self.battery.is_critically_low(),
        }

    @property
    def history(self) -> dict:
        return {
            'time':  self._history_time,
            'soc':   self._history_soc,
            'solar': self._history_solar,
            'load':  self._history_load,
        }