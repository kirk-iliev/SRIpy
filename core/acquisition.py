import time
import logging
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from core.data_model import BurstResult

class BurstWorker(QObject):
    """
    Handles high-speed burst acquisition.
    Uses a two-phase 'Acquire-First, Fit-Later' strategy to decouple 
    camera timing from CPU-intensive analysis.
    """
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
        self.logger = logging.getLogger(__name__)

    def run_burst(self):
        try:
            # Capture all raw data first to ensure consistent frame intervals.
            raw_lineouts = []
            timestamps = []
            
            # Reset stream to clear old buffers and ensure clean state
            try:
                self.driver.stop_stream()
                time.sleep(0.2)  # Give stream time to fully shut down
                self.driver.start_stream()
                time.sleep(0.1)  # Give stream time to start
            except Exception as e:
                self.logger.warning(f"Burst stream init warning: {e}")

            captured_count = 0
            for i in range(self.n_frames):
                # Timeout is generous as we are not CPU constrained here
                try:
                    img = self.driver.acquire_frame(timeout=1.0)
                except Exception as e:
                    self.logger.error(f"Failed to acquire frame {i}: {e}", exc_info=True)
                    self.error.emit(f"Frame acquisition failed at frame {i}: {str(e)}")
                    break
                
                if img is None: 
                    self.logger.warning(f"Frame {i} dropped (timeout)")
                    continue

                ts = time.time()
                
                # Minimal pre-processing (fast matrix ops only)
                img = img.squeeze().astype(np.float32, copy=False)
                # Ensure contiguous memory for in-place operations
                img = np.ascontiguousarray(img)

                if self.background is not None and img.shape == self.background.shape:
                    # Ensure background is same dtype and contiguous
                    bg = self.background.astype(np.float32, copy=False)
                    if not bg.flags['C_CONTIGUOUS']:
                        bg = np.ascontiguousarray(bg)
                    # In-place subtraction and clamp negatives to zero
                    img -= bg
                    np.clip(img, a_min=0, a_max=None, out=img)

                if self.transpose:
                    # Transpose can return non-contiguous views; make contiguous
                    img = np.ascontiguousarray(img.T)

                # Extract lineout with proper ROI handling
                # NOTE: roi_slice and roi_x_map are always in ORIGINAL (non-transposed) coordinates
                # We must swap them if transpose is enabled
                if self.transpose and self.roi_slice:
                    # After transpose: image is (orig_width, orig_height)
                    # roi_x_map (original column range) now indexes height
                    # roi_slice (original row range) now indexes width
                    row_slice = slice(int(self.roi_x_map[0]), int(self.roi_x_map[1]))
                    lineout = self.fitter.get_lineout(img, row_slice)
                elif self.roi_slice:
                    # Normal orientation
                    lineout = self.fitter.get_lineout(img, self.roi_slice)
                else:
                    lineout = self.fitter.get_lineout(img)
                
                # Apply X-ROI (width crop)
                # After transpose swap, we need to use roi_slice for column cropping
                if self.transpose and self.roi_slice:
                    # Columns are original rows
                    x_min, x_max = int(self.roi_slice.start), int(self.roi_slice.stop)
                elif self.roi_x_map:
                    # Normal orientation
                    x_min, x_max = self.roi_x_map
                else:
                    x_min, x_max = 0, len(lineout)
                
                x_min = max(0, int(x_min))
                x_max = min(len(lineout), int(x_max))
                y_fit_data = lineout[x_min:x_max] if x_max > x_min else lineout

                # Store raw data for Phase 2
                raw_lineouts.append(np.nan_to_num(y_fit_data))
                timestamps.append(ts)

                # Update progress: 0% -> 50% using captured frame count (1..n_frames)
                captured_count += 1
                pct = int(round((captured_count / float(self.n_frames)) * 50.0))
                pct = max(0, min(50, pct))
                self.progress.emit(pct)

            # Stop stream immediately to release hardware resources
            self.driver.stop_stream()
            
            # Process stored data without real-time constraints
            vis_list = []
            sigma_list = []
            total_captured = len(raw_lineouts)
            
            if total_captured == 0:
                # No frames captured; emit complete progress and finish
                self.progress.emit(100)
            else:
                processed_count = 0
                for i, y_data in enumerate(raw_lineouts):
                    res = self.fitter.fit(y_data)
                    
                    if res.success:
                        vis_list.append(res.visibility)
                        sigma_list.append(res.sigma_microns)
                    else:
                        vis_list.append(0.0)
                        sigma_list.append(0.0)

                    processed_count += 1
                    # Update progress: 50% -> 100% using processed_count
                    pct = 50 + int(round((processed_count / float(total_captured)) * 50.0))
                    pct = max(50, min(100, pct))
                    self.progress.emit(pct)

                # Ensure we end at 100%
                self.progress.emit(100)

            # Compile and emit results
            burst_res = BurstResult(
                n_frames = self.n_frames,
                mean_visibility = float(np.mean(vis_list)) if vis_list else 0.0,
                std_visibility = float(np.std(vis_list)) if vis_list else 0.0,
                mean_sigma = float(np.mean(sigma_list)) if sigma_list else 0.0,
                std_sigma = float(np.std(sigma_list)) if sigma_list else 0.0,
                vis_history = vis_list,
                sigma_history = sigma_list,
                timestamps = timestamps,
                lineout_history = raw_lineouts
            )
            self.finished.emit(burst_res)

        except Exception as e:
            self.error.emit(str(e))
            try: self.driver.stop_stream()
            except: pass