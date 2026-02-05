# core/acquisition_manager.py
import logging
import time
import numpy as np
from typing import Optional, Tuple
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from hardware.manta_driver import MantaDriver
from hardware.camera_io_thread import CameraIoThread
from core.config_manager import ConfigManager
from analysis.fitter import InterferenceFitter
from analysis.analysis_worker import AnalysisWorker

class AcquisitionManager(QObject):
    # --- Signals (Pure Data Only) ---
    live_data_ready = pyqtSignal(object, object) # Emits (image, lineout)
    fit_result_ready = pyqtSignal(object, object) # Emits (result, x_axis)
    roi_updated = pyqtSignal(float, float, float, float)  # y_min, y_max, x_min, x_max
    saturation_updated = pyqtSignal(bool)
    
    burst_progress = pyqtSignal(int)
    burst_finished = pyqtSignal(object)
    burst_error = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    # Internal wiring
    _request_fit = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # --- Internal State (Faster than reading UI) ---
        self.config_manager = ConfigManager()
        self.cfg = self.config_manager.load()
        
        self.roi_slice: slice = slice(400, 800)
        self.roi_x_limits: Tuple[int, int] = (800, 1200)
        self.autocenter_enabled = True
        self.transpose_enabled = False
        self.subtract_background = False
        self.background_frame = None
        self.saturation_threshold = 4090

        # Cached outputs
        self.last_raw_image = None
        self.last_lineout = None
        self.last_fit_result = None
        self.last_fit_x = None
        self.last_saturated = False

        # Analysis throttling
        self._analysis_busy = False
        self._last_fit_request_time = 0.0
        self._live_running = False
        self._was_live_before_burst = False
        
        # Hardware & Logic
        self.driver = MantaDriver()
        self.fitter = InterferenceFitter()
        self.fitter_burst = InterferenceFitter()
        
        # Threads
        self.camera_thread: Optional[CameraIoThread] = None
        self.an_thread: Optional[QThread] = None
        self.an_worker: Optional[AnalysisWorker] = None
        
        self._setup_analysis_thread()
        self.apply_config()

    def initialize(self):
        try:
            self.driver.connect()
        except Exception as e:
            raise RuntimeError(f"Camera connection failed: {e}")

        if self.camera_thread is None:
            self.camera_thread = CameraIoThread(self.driver, self.fitter_burst)
            self.camera_thread.frame_ready.connect(self._process_live_frame)
            self.camera_thread.background_ready.connect(self._handle_background_frame)
            self.camera_thread.burst_progress.connect(self.burst_progress.emit)
            self.camera_thread.burst_finished.connect(self._handle_burst_finished)
            self.camera_thread.burst_error.connect(self.burst_error.emit)
            self.camera_thread.error.connect(self.error_occurred.emit)
            self.camera_thread.start()

    def _setup_analysis_thread(self):
        self.an_thread = QThread()
        self.an_worker = AnalysisWorker(self.fitter)
        self.an_worker.moveToThread(self.an_thread)
        self._request_fit.connect(self.an_worker.process_fit)
        self.an_worker.result_ready.connect(self._handle_fit_result)
        self.an_thread.start()

    def apply_config(self) -> None:
        c = self.cfg
        self.roi_slice = slice(int(c['roi']['rows_min']), int(c['roi']['rows_max']))
        self.roi_x_limits = (int(c['roi']['fit_width_min']), int(c['roi']['fit_width_max']))
        self.autocenter_enabled = bool(c['roi']['auto_center'])
        self.transpose_enabled = bool(c['camera']['transpose'])
        self.saturation_threshold = int(c['camera'].get('saturation_threshold', 4090))
        min_sig = c.get('analysis', {}).get('min_signal_threshold', 50.0)
        self.fitter.min_signal = min_sig
        self.fitter_burst.min_signal = min_sig
        self.set_physics_params(
            c['physics']['wavelength_nm'] * 1e-9,
            c['physics']['slit_separation_mm'] * 1e-3,
            c['physics']['distance_m'],
        )

    # --- Fast Setters (Called by Controller) ---
    def set_roi(self, y_min, y_max, x_min, x_max):
        # Updates internal state instantly
        self.roi_slice = slice(int(y_min), int(y_max))
        self.roi_x_limits = (int(x_min), int(x_max))

    def set_autocenter(self, enabled: bool):
        self.autocenter_enabled = enabled

    def set_exposure(self, val_ms):
        if self.camera_thread is not None:
            self.camera_thread.enqueue("set_exposure", val_ms)
        if self.subtract_background:
            self.subtract_background = False # Auto-reset background

    def set_physics_params(self, wavelength, slit, distance):
        """Called by Controller to update physics constants."""
        self.fitter.wavelength = wavelength
        self.fitter.slit_sep = slit
        self.fitter.distance = distance
        
        # Sync burst fitter too so high-speed capture uses same math
        self.fitter_burst.wavelength = wavelength
        self.fitter_burst.slit_sep = slit
        self.fitter_burst.distance = distance

    def set_gain(self, val_db):
        if self.camera_thread is not None:
            self.camera_thread.enqueue("set_gain", val_db)
        if self.subtract_background:
            self.subtract_background = False

    def set_transpose(self, enabled: bool):
        self.transpose_enabled = enabled

    def toggle_background(self, enabled: bool):
        if enabled and self.background_frame is None:
            self.error_occurred.emit("No background frame available.")
            return
        self.subtract_background = enabled

    # --- Actions ---
    def capture_background(self):
        if self.camera_thread is None:
            return
        self.camera_thread.enqueue("capture_background")

    def start_live(self):
        if self.is_live_running():
            return
        if self.camera_thread is None:
            return
        self._live_running = True
        self.camera_thread.enqueue("start_live")

    def stop_live(self):
        if self.camera_thread is None:
            return
        self._live_running = False
        self.camera_thread.enqueue("stop_live")

    def is_live_running(self):
        return self._live_running

    def start_burst(self, n_frames):
        if self.camera_thread is None:
            return
        self._was_live_before_burst = self._live_running
        self._live_running = False
        self.camera_thread.enqueue(
            "start_burst",
            n_frames,
            self.roi_slice,
            self.roi_x_limits,
            self.transpose_enabled,
            self.background_frame if self.subtract_background else None,
        )

    # --- Processing Loop (Optimized) ---
    def _process_live_frame(self, raw_img):
        try:
            # 1. Fast Math
            img = raw_img.squeeze().astype(np.float32, copy=False)
            img = np.ascontiguousarray(img)
            
            if self.subtract_background and self.background_frame is not None:
                if img.shape == self.background_frame.shape:
                    img -= self.background_frame
                    np.clip(img, 0, None, out=img)

            if self.transpose_enabled:
                img = np.ascontiguousarray(img.T)

            self.last_raw_image = img
            is_saturated = np.max(img) >= self.saturation_threshold
            if is_saturated != self.last_saturated:
                self.last_saturated = is_saturated
                self.saturation_updated.emit(is_saturated)

            # 2. Cache ROI locally (No UI reads)
            # NOTE: roi_slice and roi_x_limits are stored in ORIGINAL (non-transposed) coordinates
            # We must swap them for display-space operations when transpose is enabled
            h, w = img.shape

            if self.transpose_enabled:
                # Display rows correspond to original columns
                disp_row_start, disp_row_stop = int(self.roi_x_limits[0]), int(self.roi_x_limits[1])
                # Display columns correspond to original rows
                disp_col_start, disp_col_stop = int(self.roi_slice.start), int(self.roi_slice.stop)
            else:
                disp_row_start, disp_row_stop = int(self.roi_slice.start), int(self.roi_slice.stop)
                disp_col_start, disp_col_stop = int(self.roi_x_limits[0]), int(self.roi_x_limits[1])

            # Clamp display-space ROI to image bounds
            disp_row_start = max(0, min(disp_row_start, h - 1))
            disp_row_stop = max(disp_row_start + 1, min(disp_row_stop, h))
            disp_col_start = max(0, min(disp_col_start, w - 1))
            disp_col_stop = max(disp_col_start + 1, min(disp_col_stop, w))

            # Map back to original coordinate system for storage
            if self.transpose_enabled:
                orig_row_start, orig_row_stop = disp_col_start, disp_col_stop
                orig_col_start, orig_col_stop = disp_row_start, disp_row_stop
            else:
                orig_row_start, orig_row_stop = disp_row_start, disp_row_stop
                orig_col_start, orig_col_stop = disp_col_start, disp_col_stop

            roi_changed = False
            if (orig_row_start, orig_row_stop) != (self.roi_slice.start, self.roi_slice.stop):
                self.roi_slice = slice(int(orig_row_start), int(orig_row_stop))
                roi_changed = True
            if (orig_col_start, orig_col_stop) != self.roi_x_limits:
                self.roi_x_limits = (int(orig_col_start), int(orig_col_stop))
                roi_changed = True
            if roi_changed:
                self.roi_updated.emit(
                    self.roi_slice.start,
                    self.roi_slice.stop,
                    self.roi_x_limits[0],
                    self.roi_x_limits[1],
                )

            # Extract lineout by summing the selected rows in display space
            lineout = np.sum(img[disp_row_start:disp_row_stop, :], axis=0)

            # 3. Auto-Center (Logic only, no UI updates)
            if self.autocenter_enabled:
                # FIX: Handle potential NaN/Inf in lineout just in case
                if np.all(np.isfinite(lineout)):
                    peak_idx = np.argmax(lineout)
                    # Only auto-center if we have a real signal (noise floor check)
                    if lineout[peak_idx] > 200:
                        current_w = disp_col_stop - disp_col_start
                        new_min = max(0, peak_idx - current_w // 2)
                        new_max = min(w, new_min + current_w)

                        if self.transpose_enabled:
                            # In transpose mode, lineout axis maps to original rows
                            new_rows = (int(new_min), int(new_max))
                            if new_rows != (self.roi_slice.start, self.roi_slice.stop):
                                self.roi_slice = slice(new_rows[0], new_rows[1])
                                self.roi_updated.emit(
                                    self.roi_slice.start,
                                    self.roi_slice.stop,
                                    self.roi_x_limits[0],
                                    self.roi_x_limits[1],
                                )
                        else:
                            new_cols = (int(new_min), int(new_max))
                            if new_cols != self.roi_x_limits:
                                self.roi_x_limits = new_cols
                                self.roi_updated.emit(
                                    self.roi_slice.start,
                                    self.roi_slice.stop,
                                    self.roi_x_limits[0],
                                    self.roi_x_limits[1],
                                )

            # 4. Emit Data
            self.last_lineout = lineout
            self.live_data_ready.emit(img, lineout)
            
            # 5. Analysis
            now = time.time()
            if self._analysis_busy and (now - self._last_fit_request_time) <= 3.0:
                return
            if self._analysis_busy and (now - self._last_fit_request_time) > 3.0:
                self._analysis_busy = False
            if disp_col_stop > disp_col_start:
                fit_y = lineout[disp_col_start:disp_col_stop]
                fit_x = np.arange(disp_col_start, disp_col_stop)
                self._analysis_busy = True
                self._last_fit_request_time = now
                self._request_fit.emit(fit_y, fit_x)

        except Exception as e:
            self.logger.error(f"Processing error: {e}")

    def _handle_burst_finished(self, result):
        if self._was_live_before_burst:
            self._live_running = True
        self._was_live_before_burst = False
        self.burst_finished.emit(result)

    def shutdown(self):
        self.stop_live()
        if self.an_thread:
            self.an_thread.quit()
            self.an_thread.wait(2000)
        if self.camera_thread is not None:
            self.camera_thread.enqueue("shutdown")
            self.camera_thread.wait(2000)
        if self.driver: self.driver.close()

    def _handle_background_frame(self, frame):
        if frame is None:
            return
        self.background_frame = frame.squeeze().astype(np.float32)
        self.subtract_background = True

    def _handle_fit_result(self, result, x_axis):
        self._analysis_busy = False
        self.last_fit_result = result
        self.last_fit_x = x_axis
        self.fit_result_ready.emit(result, x_axis)