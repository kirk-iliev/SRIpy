import json
import os
import copy
import logging
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """
    Handles saving and loading of application state (ROI, Camera Settings, Physics).
    """
    DEFAULT_CONFIG = {
        "camera": {
            "exposure_ms": 5.0,
            "gain_db": 0.0,
            "transpose": False,
            "subtract_background": False,
            "saturation_threshold": 4090
        },
        "analysis": {
            "min_signal_threshold": 50
        },
        "physics": {
            "wavelength_nm": 550.0,
            "slit_separation_mm": 50.0,
            "distance_m": 16.5
        },
        "roi": {
            "rows_min": 400,
            "rows_max": 800,
            "fit_width_min": 800,
            "fit_width_max": 1200,
            "auto_center": True
        }
    }

    def __init__(self, filename="sri_config.json"):
        project_root = Path(__file__).resolve().parent.parent
        self.filepath = str(project_root / filename)
        self.logger = logging.getLogger(__name__)
        # Use a deep copy to avoid shared nested structures between instances
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)

    def load(self) -> Dict[str, Any]:
        """Loads config from disk, falling back to defaults if missing/corrupt."""
        if not os.path.exists(self.filepath):
            return self.config

        try:
            with open(self.filepath, 'r') as f:
                loaded = json.load(f)
                # Deep merge to ensure new keys in defaults aren't lost
                self._deep_update(self.config, loaded)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        return self.config

    def save(self, current_state: Dict[str, Any]):
        """Writes current state to disk."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(current_state, f, indent=4)
            self.logger.info(f"Configuration saved to {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def _deep_update(self, base_dict, update_dict):
        """Recursive update for nested dictionaries."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value