import time
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from core.data_model import BurstResult

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
            
            # Ensure any previous stream is cleared
            try:
                self.driver.stop_stream()
                time.sleep(0.1) # Brief pause to let buffers clear
                self.driver.start_stream()
                time.sleep(0.1) # Allow stream to spin up
            except Exception as e:
                print(f"Burst stream init warning: {e}")

            for i in range(self.n_frames):
                # If the camera runs at 30fps, a frame takes ~33ms. 
                img = self.driver.acquire_frame(timeout=1.0)
                
                if img is None: 
                    print(f"Frame {i} dropped (timeout)")
                    continue

                # --- Fast Pre-processing ---
                img = img.squeeze().astype(np.float32)
                
                if self.background is not None and img.shape == self.background.shape:
                    img = img - self.background
                    img[img < 0] = 0
                
                if self.transpose:
                    img = img.T

                # --- Slicing ---
                if self.roi_slice:
                    lineout = self.fitter.get_lineout(img, self.roi_slice)
                else:
                    lineout = self.fitter.get_lineout(img)
                
                # Apply X-ROI (Width)
                if self.roi_x_map:
                    x_min, x_max = self.roi_x_map
                    x_min = max(0, int(x_min))
                    x_max = min(len(lineout), int(x_max))
                    y_fit_data = lineout[x_min:x_max] if x_max > x_min else lineout
                else:
                    y_fit_data = lineout

                # Store for history
                lineout_list.append(np.nan_to_num(y_fit_data)) 
                
                # --- Fitting ---
                # This is the CPU bottleneck. 
                # If this takes >30ms, it will slow down acquisition.
                res = self.fitter.fit(y_fit_data)
                
                if res.success:
                    vis_list.append(res.visibility)
                    sigma_list.append(res.sigma_microns)
                else:
                    vis_list.append(0.0)
                    sigma_list.append(0.0)
                
                timestamps.append(time.time())
                self.progress.emit(i + 1)
                
            # Cleanup
            self.driver.stop_stream()
            
            # Compile Results
            burst_res = BurstResult(
                n_frames = self.n_frames,
                mean_visibility = float(np.mean(vis_list)) if vis_list else 0.0,
                std_visibility = float(np.std(vis_list)) if vis_list else 0.0,
                mean_sigma = float(np.mean(sigma_list)) if sigma_list else 0.0,
                std_sigma = float(np.std(sigma_list)) if sigma_list else 0.0,
                vis_history = vis_list,
                sigma_history = sigma_list,
                timestamps = timestamps,
                lineout_history = lineout_list 
            )
            self.finished.emit(burst_res)

        except Exception as e:
            self.error.emit(str(e))
            # Ensure we don't leave the camera hanging
            try: self.driver.stop_stream()
            except: pass