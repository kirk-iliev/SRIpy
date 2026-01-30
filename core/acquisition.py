from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
from core.data_model import BurstResult
import time

class BurstWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object) 
    error = pyqtSignal(str)

    def __init__(self, driver, fitter, n_frames, roi_slice, roi_x_map, transpose=False, background=None):
        super().__init__()
        self.driver = driver
        self.fitter = fitter
        self.n_frames = n_frames
        self.roi_slice = roi_slice
        self.roi_x_map = roi_x_map 
        self.transpose = transpose
        self.background = background

    def run_burst(self):
        try:
            vis_list = []
            sigma_list = []
            timestamps = []
            lineout_list = [] 
            
            # Ensure fresh stream start
            self.driver.stop_stream()
            self.driver.start_stream()
            
            for i in range(self.n_frames):
                # Timeout must be generous to prevent race conditions
                img = self.driver.acquire_frame(timeout=2.0)
                if img is None: 
                    continue

                img = img.squeeze().astype(np.float32)
                
                # Must match update_frame logic exactly
                if self.background is not None and img.shape == self.background.shape:
                    img = img - self.background
                    img[img < 0] = 0
                
                if self.transpose:
                    img = img.T
                if self.roi_slice:
                    lineout = self.fitter.get_lineout(img, self.roi_slice)
                else:
                    lineout = self.fitter.get_lineout(img)
                
                if self.roi_x_map is not None:
                    x_min, x_max = self.roi_x_map
                    # Safety clamp
                    x_min = max(0, int(x_min))
                    x_max = min(len(lineout), int(x_max))
                    if x_max > x_min:
                        y_fit_data = lineout[x_min:x_max]
                    else:
                        y_fit_data = lineout
                else:
                    y_fit_data = lineout

                # Store result
                lineout_list.append(y_fit_data) 
                
                res = self.fitter.fit(y_fit_data)
                
                if res.success:
                    vis_list.append(res.visibility)
                    sigma_list.append(res.sigma_microns)
                else:
                    vis_list.append(0.0)
                    sigma_list.append(0.0)
                
                timestamps.append(time.time())
                self.progress.emit(i + 1)
                
            burst_res = BurstResult(
                n_frames = self.n_frames,
                mean_visibility = float(np.mean(vis_list)),
                std_visibility = float(np.std(vis_list)),
                mean_sigma = float(np.mean(sigma_list)),
                std_sigma = float(np.std(sigma_list)),
                vis_history = vis_list,
                sigma_history = sigma_list,
                timestamps = timestamps,
                lineout_history = lineout_list 
            )
            
            self.finished.emit(burst_res)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Clean up stream so Live View can take over later
            self.driver.stop_stream()