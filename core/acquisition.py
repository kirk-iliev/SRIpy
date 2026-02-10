import time
import logging
import queue
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from core.data_model import BurstResult

class BurstWorker(QObject):
    """
    Handles high-speed burst processing.
    Receives frames from a queue (CameraIoThread is sole driver owner).
    Uses a two-phase 'Acquire-First, Fit-Later' strategy to decouple 
    camera timing from CPU-intensive analysis.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(object) 
    error = pyqtSignal(str)

    def __init__(self, frame_queue: queue.Queue, fitter, n_frames, roi_slice, roi_x_map, transpose=False, background=None):
        super().__init__()
        self.frame_queue = frame_queue  # Receives frames from CameraIoThread
        self.fitter = fitter
        self.n_frames = n_frames
        self.roi_slice = roi_slice
        self.roi_x_map = roi_x_map
        self.transpose = transpose
        self.background = background
        self.logger = logging.getLogger(__name__)

    def run_burst(self):
        try:
            # Phase 1: Collect all frames from the queue (CameraIoThread pushes them)
            raw_lineouts = []
            timestamps = []
            
            self.logger.info(f"Burst: Waiting for {self.n_frames} frames from camera thread")
            
            captured_count = 0
            for i in range(self.n_frames):
                # Timeout is generous as we wait for camera thread to deliver frames
                try:
                    img = self.frame_queue.get(block=True, timeout=2.0)
                except queue.Empty:
                    self.logger.warning(f"Frame {i} timeout waiting for camera thread")
                    self.error.emit(f"Frame acquisition timeout at frame {i}")
                    break
                
                if img is None: 
                    self.logger.warning(f"Frame {i} dropped (sentinel)")
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
                
                # Apply X-ROI
                # After transpose swap, use roi_slice for column cropping
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

            # Phase 2: Process stored data without real-time constraints
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
            self.logger.error(f"Burst error: {e}", exc_info=True)
            self.error.emit(str(e))