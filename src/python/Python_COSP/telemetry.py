"""
telemetry.py - Telemetry subsystem for CubeSat.

Collects housekeeping and mission data, formats TM packets,
simulates UHF downlink windows, and stores a full mission log.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import time as _time


# ── Telemetry packet ──────────────────────────────────────────────────────────

@dataclass
class TelemetryPacket:
    """
    CCSDS-style housekeeping telemetry packet.

    All physical values stored in SI units internally;
    human-readable strings formatted for display.
    """
    mission_time:     float          # [s]  elapsed since launch
    utc_timestamp:    str            # ISO-8601 string

    # Navigation
    position_x:       float          # [m]
    position_y:       float          # [m]
    velocity_x:       float          # [m/s]
    velocity_y:       float          # [m/s]
    altitude:         float          # [m]
    speed:            float          # [m/s]

    # Orbital elements
    semi_major_axis:  float          # [m]
    eccentricity:     float
    periapsis:        float          # [m]
    apoapsis:         float          # [m]
    orbital_period:   float          # [s]

    # Power
    battery_soc:      float          # [0–1]
    solar_power_W:    float          # [W]
    load_power_W:     float          # [W]
    in_shadow:        bool

    # ADCS
    attitude_angle:   float          # [rad]

    # Mission
    phase:            str
    mass_kg:          float          # [kg]

    # Optional camera flag
    imaging_active:   bool = False

    def to_dict(self) -> dict:
        return {
            'mission_time':    round(self.mission_time,    2),
            'utc':             self.utc_timestamp,
            'pos_x_km':        round(self.position_x / 1e3, 3),
            'pos_y_km':        round(self.position_y / 1e3, 3),
            'vel_x_ms':        round(self.velocity_x,       2),
            'vel_y_ms':        round(self.velocity_y,       2),
            'altitude_km':     round(self.altitude   / 1e3, 3),
            'speed_kms':       round(self.speed      / 1e3, 4),
            'sma_km':          round(self.semi_major_axis / 1e3, 2),
            'eccentricity':    round(self.eccentricity, 6),
            'periapsis_km':    round(self.periapsis / 1e3, 2),
            'apoapsis_km':     round(self.apoapsis  / 1e3, 2),
            'period_min':      round(self.orbital_period / 60.0, 2),
            'battery_pct':     round(self.battery_soc * 100.0, 1),
            'solar_W':         round(self.solar_power_W, 2),
            'load_W':          round(self.load_power_W,  2),
            'in_shadow':       self.in_shadow,
            'attitude_deg':    round(np.degrees(self.attitude_angle), 2),
            'phase':           self.phase,
            'mass_kg':         round(self.mass_kg, 3),
            'imaging_active':  self.imaging_active,
        }

    def format_display(self) -> str:
        """Human-readable summary for console / HUD."""
        d = self.to_dict()
        lines = [
            f"T+{d['mission_time']:8.1f}s  [{d['phase']}]",
            f"  Alt: {d['altitude_km']:8.2f} km    Speed: {d['speed_kms']:.4f} km/s",
            f"  SMA: {d['sma_km']:8.2f} km    Ecc:   {d['eccentricity']:.6f}",
            f"  Pe:  {d['periapsis_km']:8.2f} km    Ap:    {d['apoapsis_km']:.2f} km",
            f"  Batt:{d['battery_pct']:5.1f}%        Solar: {d['solar_W']:.1f} W",
            f"  {'[SHADOW]' if d['in_shadow'] else '[SUNLIT]'}",
        ]
        return "\n".join(lines)


# ── Telemetry subsystem ───────────────────────────────────────────────────────

class TelemetrySystem:
    """
    Manages telemetry collection, downlink simulation, and mission logging.

    Downlink window simulation:
      UHF at 9600 baud → ~1.2 kB/s effective.
      Ground contact assumed when elevation > 5° (simplified: every ~90 min
      we get a ~10-minute pass, modelled as a continuous flag here).
    """

    DOWNLINK_RATE_BPS = 9600        # UHF raw bit rate
    PACKET_SIZE_BYTES = 256         # housekeeping packet size
    PACKET_INTERVAL_S = 5.0         # how often we generate a TM packet

    def __init__(self):
        self._packets:      List[TelemetryPacket] = []
        self._downlinked:   List[TelemetryPacket] = []
        self._pending_queue: List[TelemetryPacket] = []

        self._last_packet_time: float = -999.0
        self._ground_contact:   bool  = False
        self._total_bytes_down: int   = 0

        # Mission event log
        self._events: List[Dict[str, Any]] = []

        # Anomaly counters
        self._anomaly_count = 0

        # Start real-clock reference
        self._wall_clock_start = _time.time()

    # ── Packet generation ─────────────────────────────────────────────────────

    def collect(self,
                mission_time: float,
                position:     np.ndarray,
                velocity:     np.ndarray,
                orbital_elements: dict,
                power_state:  dict,
                phase:        str,
                mass_kg:      float,
                attitude:     float = 0.0,
                imaging:      bool  = False) -> TelemetryPacket | None:
        """
        Attempt to generate a new TM packet.
        Returns a packet if the collection interval has elapsed, else None.
        """
        if mission_time - self._last_packet_time < self.PACKET_INTERVAL_S:
            return None

        self._last_packet_time = mission_time

        # UTC-like timestamp (relative to epoch)
        utc_str = f"T+{mission_time/3600:.4f}h"

        pkt = TelemetryPacket(
            mission_time   = mission_time,
            utc_timestamp  = utc_str,
            position_x     = float(position[0]),
            position_y     = float(position[1]),
            velocity_x     = float(velocity[0]),
            velocity_y     = float(velocity[1]),
            altitude       = float(orbital_elements.get('altitude', 0.0)),
            speed          = float(np.linalg.norm(velocity)),
            semi_major_axis= float(orbital_elements.get('semi_major_axis', 0.0)),
            eccentricity   = float(orbital_elements.get('eccentricity', 0.0)),
            periapsis      = float(orbital_elements.get('periapsis', 0.0)),
            apoapsis       = float(orbital_elements.get('apoapsis', 0.0)),
            orbital_period = float(orbital_elements.get('orbital_period', 0.0)),
            battery_soc    = float(power_state.get('battery_soc', 0.0)),
            solar_power_W  = float(power_state.get('solar_power_W', 0.0)),
            load_power_W   = float(power_state.get('load_power_W', 0.0)),
            in_shadow      = bool(power_state.get('in_shadow', False)),
            attitude_angle = float(attitude),
            phase          = str(phase),
            mass_kg        = float(mass_kg),
            imaging_active = bool(imaging),
        )

        self._packets.append(pkt)
        self._pending_queue.append(pkt)
        return pkt

    # ── Ground contact / downlink ─────────────────────────────────────────────

    def update_downlink(self, mission_time: float, dt: float):
        """
        Simulate ground contact and downlink pending packets.
        Simplified: contact window every 5700 s (≈95-min orbit) for 600 s.
        """
        orbit_period = 5700.0
        contact_duration = 600.0
        phase_in_orbit = mission_time % orbit_period
        self._ground_contact = phase_in_orbit < contact_duration

        if self._ground_contact and self._pending_queue:
            # Drain queue at downlink rate
            max_packets_per_step = max(1, int(
                (self.DOWNLINK_RATE_BPS / 8) * dt / self.PACKET_SIZE_BYTES
            ))
            for _ in range(min(max_packets_per_step, len(self._pending_queue))):
                pkt = self._pending_queue.pop(0)
                self._downlinked.append(pkt)
                self._total_bytes_down += self.PACKET_SIZE_BYTES

    # ── Event logging ─────────────────────────────────────────────────────────

    def log_event(self, mission_time: float, event_type: str, description: str, data: dict = None):
        self._events.append({
            'time':        mission_time,
            'type':        event_type,
            'description': description,
            'data':        data or {},
        })
        print(f"  [TM EVENT T+{mission_time:7.1f}s] {event_type}: {description}")

    def log_anomaly(self, mission_time: float, description: str):
        self._anomaly_count += 1
        self.log_event(mission_time, 'ANOMALY', description)

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        return {
            'total_packets':     len(self._packets),
            'downlinked_packets':len(self._downlinked),
            'pending_packets':   len(self._pending_queue),
            'total_bytes_down':  self._total_bytes_down,
            'anomaly_count':     self._anomaly_count,
            'ground_contact':    self._ground_contact,
            'events':            self._events,
        }

    def save_log(self, path: str = 'telemetry_log.json'):
        """Export full telemetry log to JSON."""
        payload = {
            'summary': self.get_summary(),
            'packets': [p.to_dict() for p in self._packets],
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[TM] Telemetry log saved → {path}  ({len(self._packets)} packets)")

    @property
    def packets(self) -> List[TelemetryPacket]:
        return self._packets

    @property
    def ground_contact(self) -> bool:
        return self._ground_contact

    @property
    def latest(self) -> TelemetryPacket | None:
        return self._packets[-1] if self._packets else None