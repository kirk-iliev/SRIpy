from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class ExperimentSettings:
    """Settings for experimental setup."""

    # Cam settings
    shutter_time_s: float = 0.001 # 1ms default
    gain_db: float = 0.0
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 1024
    roi_height: int = 768

    # Physical settings
    slit_separation_m: float = 0.05
    wavelength_m: float = 550e-9
    distance_L_m: float = 16.5
    pixel_scale_um: float = 3.75

@dataclass
class ExperimentData: 

    raw_image: np.ndarray = field(default_factory=lambda: np.zeros((100, 100)))
    timestamp: float = field(default_factory=time.time)

    # Processed data
    profile_1d: np.ndarray = None
    visibility: float = None
    