from dataclasses import dataclass, asdict, field
import numpy as np
import json
import os
import scipy.io
from datetime import datetime, timezone

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, datetime):
            # Ensure timezone-aware datetimes are serialized to ISO format
            return o.isoformat()
        return super(NumpyEncoder, self).default(o)

@dataclass
class ExperimentMetadata:
    exposure_s: float = 0.005
    gain_db: float = 0.0
    wavelength_nm: float = 550.0
    slit_separation_mm: float = 50.0
    distance_m: float = 16.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ExperimentResult:
    visibility: float = 0.0
    sigma_microns: float = 0.0
    lineout_x: list = field(default_factory=list)
    lineout_y: list = field(default_factory=list)
    fit_y: list = field(default_factory=list)
    is_saturated: bool = False

@dataclass
class BurstResult:
    """Stores statistics from a multi-frame acquisition."""
    n_frames: int = 0
    mean_visibility: float = 0.0
    std_visibility: float = 0.0
    mean_sigma: float = 0.0
    std_sigma: float = 0.0
    
    # History Arrays
    vis_history: list = field(default_factory=list)
    sigma_history: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    
    # This allows you to 'replay' the burst later
    lineout_history: list = field(default_factory=list)

class DataManager:
    @staticmethod
    def save_dataset(directory, filename_prefix, raw_image, metadata: ExperimentMetadata, result: ExperimentResult):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_name = f"{filename_prefix}_{timestamp}"
        save_path = os.path.join(directory, full_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        np.save(os.path.join(save_path, "raw_image.npy"), raw_image)
        
        # Ensure timestamp reflects the save time (UTC)
        metadata.timestamp = datetime.now(timezone.utc)

        data_dict = {
            "metadata": asdict(metadata),
            "results": asdict(result)
        }
        
        json_path = os.path.join(save_path, "experiment_data.json")
        with open(json_path, 'w') as f:
            json.dump(data_dict, f, cls=NumpyEncoder, indent=4)
            
        return save_path

    @staticmethod
    def save_matlab(directory, filename_prefix, raw_image, metadata: ExperimentMetadata, result: ExperimentResult):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.mat"
        save_path = os.path.join(directory, filename)
        
        # Ensure timestamp reflects the save time (UTC)
        metadata.timestamp = datetime.now(timezone.utc)

        # scipy.io.savemat can't handle datetime objects; convert to ISO string
        meta_for_mat = asdict(metadata)
        if isinstance(meta_for_mat.get('timestamp'), datetime):
            meta_for_mat['timestamp'] = meta_for_mat['timestamp'].isoformat()

        mat_dict = {
            "IMG": {
                "raw": raw_image,
                "lineout": np.array(result.lineout_y),
                "fit_curve": np.array(result.fit_y),
                "visibility": result.visibility,
                "sigma_microns": result.sigma_microns,
                "issaturated": result.is_saturated
            },
            "META": meta_for_mat
        }
        
        scipy.io.savemat(save_path, mat_dict)
        return save_path

    @staticmethod
    def save_burst(directory, filename_prefix, burst_result: BurstResult, metadata: ExperimentMetadata):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_BURST_{timestamp}.json"
        save_path = os.path.join(directory, filename)
        
        # Ensure timestamp reflects the save time (UTC)
        metadata.timestamp = datetime.now(timezone.utc)

        data_dict = {
            "metadata": asdict(metadata),
            "burst_data": asdict(burst_result)
        }
        
        with open(save_path, 'w') as f:
            json.dump(data_dict, f, cls=NumpyEncoder, indent=4)
        return save_path