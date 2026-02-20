import time
import logging
import queue
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from core.data_model import BurstResult
from utils.image_utils import process_roi_lineout

class BurstWorker(QObject):
    """
    Worker for high-speed burst acquisition and post-process analysis.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, frame_queue: queue.Queue, fitter, n_frames, roi_slice, roi_x_map, transpose=False, background=None):
        super().__init__()
        self.frame_queue = frame_queue
        self.fitter = fitter
        self.n_frames = n_frames
        self.roi_slice = roi_slice      # Integration slice (rows)
        self.roi_x_map = roi_x_map      # Measurement limits (columns/width)
        self.transpose = transpose
        self.background = background
        self.logger = logging.getLogger(__name__)

    def run_burst(self):
        try:
            # --- Phase 1: Acquisition ---
            raw_lineouts = []
            timestamps = []
            self.logger.info(f"Burst: Waiting for {self.n_frames} frames")

            captured_count = 0
            for i in range(self.n_frames):
                try:
                    img = self.frame_queue.get(block=True, timeout=2.0)
                except queue.Empty:
                    self.logger.warning(f"Frame {i} timeout")
                    self.error.emit(f"Acquisition timeout at frame {i}")
                    break

                if img is None: continue
                ts = time.time()

                # Ensure img is a proper numpy array (may come as Vimba object)
                try:
                    if not isinstance(img, np.ndarray):
                        img = np.asarray(img, dtype=np.uint16)
                    else:
                        # Ensure it's a copy and proper dtype if needed
                        img = np.asarray(img, dtype=np.uint16)
                except Exception as e:
                    self.logger.warning(f"Frame {i} conversion error: {e}")
                    continue

                # Use shared logic to get full lineout
                _, full_lineout, _ = process_roi_lineout(
                    img, self.roi_slice, self.transpose, self.background
                )

                # Crop to measurement limits
                x_start = int(self.roi_x_map[0])
                x_stop = int(self.roi_x_map[1])
                x_start = max(0, min(x_start, len(full_lineout)))
                x_stop = max(x_start, min(x_stop, len(full_lineout)))

                raw_lineouts.append(np.nan_to_num(full_lineout[x_start:x_stop]))
                timestamps.append(ts)

                captured_count += 1
                self.progress.emit(int((captured_count / self.n_frames) * 50))

            # --- Phase 2: Analysis ---
            vis_list = []
            sigma_list = []
            total_captured = len(raw_lineouts)
            raw_vis_list = []
            max_int_list = []
            min_int_list = []

            if total_captured == 0:
                self.progress.emit(100)
                self.finished.emit(BurstResult())
                return

            for i, y_data in enumerate(raw_lineouts):
                res = self.fitter.fit(y_data)

                if res.success:
                    vis_list.append(res.visibility)
                    sigma_list.append(res.sigma_microns)
                    raw_vis_list.append(res.raw_visibility)
                    max_int_list.append(res.max_intensity)
                    min_int_list.append(res.min_intensity)
                else:
                    vis_list.append(0.0)
                    sigma_list.append(0.0)
                    raw_vis_list.append(0.0)
                    max_int_list.append(0.0)
                    min_int_list.append(0.0)

                pct = 50 + int(((i + 1) / total_captured) * 50)
                self.progress.emit(max(50, min(100, pct)))

            burst_res = BurstResult(
                n_frames=self.n_frames,
                mean_visibility=float(np.mean(vis_list)) if vis_list else 0.0,
                std_visibility=float(np.std(vis_list)) if vis_list else 0.0,
                mean_sigma=float(np.mean(sigma_list)) if sigma_list else 0.0,
                std_sigma=float(np.std(sigma_list)) if sigma_list else 0.0,
                vis_history=vis_list,
                sigma_history=sigma_list,
                timestamps=timestamps,
                lineout_history=raw_lineouts,
                mean_raw_visibility=float(np.mean(raw_vis_list)) if raw_vis_list else 0.0,
                mean_max_intensity=float(np.mean(max_int_list)) if max_int_list else 0.0,
                mean_min_intensity=float(np.mean(min_int_list)) if min_int_list else 0.0
            )
            self.finished.emit(burst_res)

        except Exception as e:
            self.logger.error(f"Burst error: {e}", exc_info=True)
            self.error.emit(str(e))