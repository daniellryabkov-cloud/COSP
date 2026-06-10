"""
camera.py - Imaging camera subsystem for CubeSat.

Simulates a miniature Earth-observation camera:
  • Ground sampling distance (GSD) calculation
  • Coverage / swath width
  • Imaging schedule (nadir-pointed passes)
  • Image metadata generation
  • Data volume accounting
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from physics import R_EARTH


@dataclass
class ImageFrame:
    """Metadata for a single captured image frame."""
    frame_id:         int
    mission_time:     float          # [s]
    altitude_m:       float          # [m]
    ground_track_lat: float          # [deg]  (simplified from 2-D position angle)
    ground_track_lon: float          # [deg]  (incremented by time)
    gsd_m:            float          # ground sampling distance [m/px]
    swath_km:         float          # swath width [km]
    exposure_ms:      float          # [ms]
    snr_db:           float          # signal-to-noise ratio estimate [dB]
    cloud_cover:      float          # 0–1 random cloud fraction
    data_size_MB:     float          # compressed image data [MB]


class Camera:
    """
    Miniature pushbroom / frame camera for 6U CubeSat.

    Optical parameters (representative of a 1U camera module):
    ──────────────────────────────────────────────────────────
    • Focal length   : 40 mm
    • Aperture       : f/4
    • Sensor         : 2048 × 1536 px,  3.45 µm pixel pitch
    • Compression    : JPEG-like, ~4:1 ratio
    • Data rate      : 8 Mpx/frame × 3 channels × 8 bits ≈ 19 MB raw → ~4.8 MB compressed
    """

    FOCAL_LENGTH_MM    = 40.0
    PIXEL_PITCH_UM     = 3.45
    SENSOR_WIDTH_PX    = 2048
    SENSOR_HEIGHT_PX   = 1536
    COMPRESSION_RATIO  = 4.0
    FRAME_RATE_HZ      = 0.1          # one frame every 10 s when imaging
    POWER_W            = 1.5          # power consumption while imaging
    READ_NOISE_E       = 6.0          # read noise electrons
    FULL_WELL_E        = 15_000.0     # full-well capacity electrons
    QUANTUM_EFF        = 0.65         # peak QE

    def __init__(self):
        self._frames:          List[ImageFrame] = []
        self._frame_id:        int   = 0
        self._is_imaging:      bool  = False
        self._total_data_MB:   float = 0.0
        self._last_frame_time: float = -999.0
        self._storage_used_MB: float = 0.0
        self._storage_total_MB: float = 2048.0   # 2 GB onboard NAND

    # ── Optical calculations ──────────────────────────────────────────────────

    def ground_sampling_distance(self, altitude_m: float) -> float:
        """
        GSD = pixel_pitch × altitude / focal_length  [m/px]
        """
        return (self.PIXEL_PITCH_UM * 1e-6) * altitude_m / (self.FOCAL_LENGTH_MM * 1e-3)

    def swath_width_km(self, altitude_m: float) -> float:
        """
        Swath = GSD × sensor_width_px  [km]
        """
        gsd = self.ground_sampling_distance(altitude_m)
        return gsd * self.SENSOR_WIDTH_PX / 1000.0

    def image_size_MB(self) -> float:
        """Compressed image file size [MB]."""
        raw_bits = self.SENSOR_WIDTH_PX * self.SENSOR_HEIGHT_PX * 3 * 8
        return raw_bits / 8.0 / 1e6 / self.COMPRESSION_RATIO

    def estimate_snr(self, altitude_m: float, sun_zenith_deg: float) -> float:
        """
        Simplified radiometric SNR estimate [dB].
        Uses surface reflectance ≈ 0.3 (vegetated land), Lambertian model.
        """
        gsd = self.ground_sampling_distance(altitude_m)
        # Ground-level solar irradiance (approx)
        irr = 1361.0 * max(0.0, np.cos(np.radians(sun_zenith_deg)))
        # Radiance at sensor
        reflectance  = 0.30
        L_ground     = reflectance * irr / np.pi                    # W/(m²·sr)
        # Solid angle of pixel projected on ground
        pixel_area   = gsd**2                                        # m²
        # Photon flux at detector (very simplified)
        h_planck     = 6.626e-34
        c_light      = 3e8
        lambda_eff   = 550e-9                                        # green band
        E_photon     = h_planck * c_light / lambda_eff
        aperture_area = np.pi * ((self.FOCAL_LENGTH_MM * 1e-3 / (2 * 4.0))**2)  # f/4
        signal_e     = (L_ground * pixel_area * aperture_area
                        * self.QUANTUM_EFF * 0.010               # 10 ms exposure
                        / E_photon)
        signal_e     = float(np.clip(signal_e, 1.0, self.FULL_WELL_E))
        noise_e      = np.sqrt(signal_e + self.READ_NOISE_E**2)
        snr_linear   = signal_e / noise_e
        return float(20.0 * np.log10(max(snr_linear, 1e-3)))

    # ── Control ───────────────────────────────────────────────────────────────

    def start_imaging(self):
        self._is_imaging = True

    def stop_imaging(self):
        self._is_imaging = False

    @property
    def is_imaging(self) -> bool:
        return self._is_imaging

    @property
    def storage_fill_fraction(self) -> float:
        return self._storage_used_MB / self._storage_total_MB

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, mission_time: float,
               position: np.ndarray,
               altitude_m: float,
               in_shadow: bool) -> Optional[ImageFrame]:
        """
        Attempt to capture a frame if imaging is active and timing allows.

        Parameters
        ----------
        mission_time : float  [s]
        position     : np.ndarray  [x, y] in metres
        altitude_m   : float  [m]
        in_shadow    : bool

        Returns
        -------
        ImageFrame if a new frame was captured, else None.
        """
        if not self._is_imaging:
            return None
        if in_shadow:
            return None
        if mission_time - self._last_frame_time < (1.0 / self.FRAME_RATE_HZ):
            return None
        if self._storage_used_MB >= self._storage_total_MB * 0.95:
            return None   # storage nearly full

        self._last_frame_time = mission_time

        # Derive ground track coordinates from position angle
        angle_rad      = np.arctan2(position[1], position[0])
        lat_approx     = np.degrees(angle_rad) % 90.0     # crude
        lon_approx     = (mission_time / 86400.0 * 360.0) % 360.0

        gsd    = self.ground_sampling_distance(altitude_m)
        swath  = self.swath_width_km(altitude_m)
        size   = self.image_size_MB()

        # Sun zenith (simplified): random 30–60° when sunlit
        sun_zen = np.random.uniform(30.0, 60.0)
        snr     = self.estimate_snr(altitude_m, sun_zen)
        cloud   = float(np.random.beta(2, 5))   # most images partly cloudy

        frame = ImageFrame(
            frame_id        = self._frame_id,
            mission_time    = mission_time,
            altitude_m      = altitude_m,
            ground_track_lat= lat_approx,
            ground_track_lon= lon_approx,
            gsd_m           = gsd,
            swath_km        = swath,
            exposure_ms     = 10.0,
            snr_db          = snr,
            cloud_cover     = cloud,
            data_size_MB    = size,
        )

        self._frames.append(frame)
        self._frame_id       += 1
        self._total_data_MB  += size
        self._storage_used_MB += size
        return frame

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        last = self._frames[-1] if self._frames else None
        return {
            'is_imaging':      self._is_imaging,
            'total_frames':    self._frame_id,
            'total_data_MB':   round(self._total_data_MB, 2),
            'storage_used_MB': round(self._storage_used_MB, 2),
            'storage_pct':     round(self.storage_fill_fraction * 100, 1),
            'last_gsd_m':      round(last.gsd_m, 1) if last else None,
            'last_swath_km':   round(last.swath_km, 1) if last else None,
            'last_snr_db':     round(last.snr_db, 1) if last else None,
        }

    @property
    def frames(self) -> List[ImageFrame]:
        return self._frames