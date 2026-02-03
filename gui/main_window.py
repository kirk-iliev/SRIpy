import sys
import os
import numpy as np
import logging
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from gui.widgets.live_monitor import LiveMonitorWidget
from gui.widgets.history_widget import HistoryWidget
from gui.widgets.control_panel import ControlPanelWidget

# --- Import Logic & Workers ---
from hardware.manta_driver import MantaDriver
from analysis.fitter import InterferenceFitter
from hardware.camera_thread import CameraWorker
from core.config_manager import ConfigManager
from core.data_model import DataManager, ExperimentMetadata, ExperimentResult

from analysis.analysis_worker import AnalysisWorker
from core.acquisition import BurstWorker

class InterferometerGUI(QMainWindow):
    request_fit = pyqtSignal(object, object)
    start_worker_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRIpy Interferometer Monitor")
        self.resize(1300, 950)

        # Init Logic & Config
        self.config_manager = ConfigManager()
        self.cfg = self.config_manager.load()
        self.fitter = InterferenceFitter()
        self.background_frame = None
        self.user_is_interacting = False
        self.worker_is_busy = False
        self.logger = logging.getLogger(__name__)

        # Init Hardware
        try:
            self.driver = MantaDriver()
            self.driver.connect()
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Could not connect:\n{e}")
            sys.exit(1)

        # Setup UI Components
        self.setup_ui()

        # Setup Threads
        self.setup_threads()

        # Apply Config to UI
        self.apply_config()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: Tabs
        from PyQt6.QtWidgets import QTabWidget
        self.tabs = QTabWidget()
        self.live_widget = LiveMonitorWidget()
        self.history_widget = HistoryWidget()
        self.tabs.addTab(self.live_widget, "Live Monitor")
        self.tabs.addTab(self.history_widget, "Stability History")
        
        # Right: Controls
        self.controls = ControlPanelWidget()

        layout.addWidget(self.tabs, stretch=4)
        layout.addWidget(self.controls, stretch=1)

        # --- Connect Signals (View -> Controller) ---
        
        # Live Monitor Signals
        self.live_widget.roi_drag_start.connect(lambda: setattr(self, 'user_is_interacting', True))
        self.live_widget.roi_drag_end.connect(lambda: setattr(self, 'user_is_interacting', False))
        
        # Control Panel Signals
        self.controls.exposure_changed.connect(self.update_exposure)
        self.controls.gain_changed.connect(self.update_gain)
        self.controls.physics_changed.connect(self.update_physics)
        
        # Buttons
        self.controls.acquire_bg_clicked.connect(self.acquire_background)
        self.controls.toggle_live_clicked.connect(self.toggle_live_view)
        self.controls.burst_clicked.connect(self.start_burst_acquisition)
        self.controls.save_data_clicked.connect(self.save_full_dataset)
        self.controls.save_mat_clicked.connect(self.save_mat_file)
        self.controls.reset_roi_clicked.connect(self.reset_rois)

    def setup_threads(self):
        # Analysis Thread
        self.an_thread = QThread()
        self.an_worker = AnalysisWorker(self.fitter)
        self.an_worker.moveToThread(self.an_thread)
        self.request_fit.connect(self.an_worker.process_fit)
        self.an_worker.result_ready.connect(self.handle_fit_result)
        self.an_thread.start()

        # Camera Thread
        self.cam_thread = QThread()
        self.cam_worker = CameraWorker(self.driver)
        self.cam_worker.moveToThread(self.cam_thread)
        self.cam_worker.frame_ready.connect(self.update_frame)
        self.start_worker_signal.connect(self.cam_worker.start_acquire)
        self.cam_thread.start()

    # --- Controller Logic ---

    def update_frame(self, img_int):
        """Main loop: Get frame -> Update Display -> Trigger Analysis"""
        try:
            if img_int is None: return
            
# Convert and ensure contiguous for efficient ops and rendering
            img = img_int.squeeze().astype(np.float32, copy=False)
            img = np.ascontiguousarray(img)
            self.current_raw_image = img

            # Check Saturation
            is_saturated = np.max(img) >= 4090

            # Background Subtraction
            if self.controls.chk_bg.isChecked() and self.background_frame is not None:
                if img.shape == self.background_frame.shape:
                    bg = self.background_frame.astype(np.float32, copy=False)
                    if not bg.flags['C_CONTIGUOUS']:
                        bg = np.ascontiguousarray(bg)
                    img -= bg
                    np.clip(img, 0, None, out=img)

            # Transpose (ensure contiguous)
            if self.controls.chk_transpose.isChecked():
                img = np.ascontiguousarray(img.T)

            # Update Visuals (provide contiguous transpose for pyqtgraph)
            self.live_widget.update_image(np.ascontiguousarray(img.T))

            # ROI Bounds Check
            h, w = img.shape
            self.ensure_roi_bounds(h, w)
            
            # Extract Lineout
            y_min, y_max = map(int, self.live_widget.get_roi_rows())
            # Clamp to image height
            y_min = max(0, min(y_min, h - 1))
            y_max = max(1, min(y_max, h))
            
            if y_max > y_min:
                roi_slice = slice(y_min, y_max)
                lineout = np.sum(img[roi_slice, :], axis=0)
            else:
                lineout = np.zeros(w)
            
            self.current_lineout = lineout
            # Provide explicit x-data so plots align with fit
            self.live_widget.update_lineout(np.arange(len(lineout)), lineout)

            # Auto-Center Logic
            if self.controls.chk_autocenter.isChecked() and not self.user_is_interacting:
                self.run_autocenter(lineout, w)

            # Trigger Analysis
            if not self.worker_is_busy:
                x_min, x_max = map(int, self.live_widget.get_roi_width())
                x_min, x_max = max(0, x_min), min(w, x_max)
                if x_max > x_min:
                    self.worker_is_busy = True
                    self.request_fit.emit(lineout[x_min:x_max], np.arange(x_min, x_max))
            
            # Update saturation status in UI
            if is_saturated:
                self.controls.lbl_sat.setText("SATURATED!")
                self.controls.lbl_sat.setStyleSheet("color: red; font-weight: bold; background-color: yellow;")
            else:
                self.controls.lbl_sat.setText("OK")
                self.controls.lbl_sat.setStyleSheet("color: green; font-weight: bold;")

        except Exception as e:
            self.logger.exception("Update error")

    def handle_fit_result(self, res, x_data):
        self.worker_is_busy = False
        self.last_fit_result = res
        
        if res.success:
            self.live_widget.update_fit(x_data, res.fitted_curve)
            self.history_widget.add_point(res.sigma_microns)
            # Update stats text
            is_sat = (self.controls.lbl_sat.text() == "SATURATED!")
            self.controls.update_stats(res.visibility, res.sigma_microns, is_sat)
        else:
            self.live_widget.update_fit([], [])

    def acquire_background(self):
        """Pauses stream safely, grabs a frame, saves as background, then restarts."""
        was_live = self.cam_worker._is_running # Check flag before we stop it

        try:
            if was_live:
                self.controls.btn_live.setChecked(False) # Update UI
                self.cam_worker.stop_acquire()           # Tell loop to break
                
                self.cam_thread.quit()
                
                # Wait up to 2 seconds for it to fully release the driver
                if not self.cam_thread.wait(2000):
                    self.logger.warning("Camera thread stuck. Forcing termination...")
                    self.cam_thread.terminate()
                    self.cam_thread.wait()

            self.driver.stop_stream() 
            
            raw = self.driver.acquire_frame(timeout=2.0)
            
            if raw is not None:
                self.background_frame = raw.squeeze().astype(np.float32)
                self.controls.chk_bg.setEnabled(True)
                self.controls.chk_bg.setChecked(True)
                self.controls.chk_bg.setText("Subtract Background (Active)")
            else:
                QMessageBox.warning(self, "Warning", "Failed to capture background frame.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Background acquisition failed: {e}")

        finally:
            if was_live:
                self.cam_thread.start() 
                
                self.controls.btn_live.setChecked(True)
                # self.toggle_live_view(True)

    def start_burst_acquisition(self):
        if self.cam_worker._is_running:
            self.controls.btn_live.setChecked(False)
            self.cam_worker.stop_acquire()

            self.cam_thread.quit()
            
            if not self.cam_thread.wait(1000):
                self.logger.warning("Camera thread did not stop cleanly. Forcing...")
                self.cam_thread.terminate()
                self.cam_thread.wait()
            
            # self.cam_thread.start() 
        
        y_min, y_max = map(int, self.live_widget.get_roi_rows())
        x_min, x_max = map(int, self.live_widget.get_roi_width())
        roi_slice = slice(y_min, y_max)
        
        bg = self.background_frame if self.controls.chk_bg.isChecked() else None
        trans = self.controls.chk_transpose.isChecked()

        n_frames = 50

        self.burst_thread = QThread()
        self.burst_worker = BurstWorker(self.driver, self.fitter, n_frames, roi_slice, (x_min, x_max), 
                                      transpose=trans, background=bg)
        self.burst_worker.moveToThread(self.burst_thread)
        
        self.burst_thread.started.connect(self.burst_worker.run_burst)
        self.burst_worker.progress.connect(self.controls.progress_bar.setValue)
        self.burst_worker.finished.connect(self.handle_burst_finished)
        self.burst_worker.error.connect(self.handle_burst_error)
        
        self.burst_worker.finished.connect(self.burst_thread.quit)
        self.burst_worker.finished.connect(self.burst_worker.deleteLater)
        self.burst_thread.finished.connect(self.burst_thread.deleteLater)
        
        # Worker now emits progress as percentage (0-100)
        self.controls.progress_bar.setRange(0, 100)
        self.controls.progress_bar.setVisible(True)
        self.controls.progress_bar.setValue(0)
        self.controls.btn_burst.setEnabled(False)
        self.controls.btn_live.setEnabled(False)
        
        self.burst_thread.start()

    def handle_burst_finished(self, res):
        self.controls.btn_burst.setEnabled(True)
        self.controls.btn_live.setEnabled(True)
        self.controls.progress_bar.setVisible(False)
        self.last_burst_result = res
        
        QMessageBox.information(self, "Burst Done", 
                              f"Mean Sigma: {res.mean_sigma:.2f} um\nMean Vis: {res.mean_visibility:.3f}")
        
        self.history_widget.add_point(res.mean_sigma)

    def handle_burst_error(self, err):
        self.controls.btn_burst.setEnabled(True)
        self.controls.btn_live.setEnabled(True)
        QMessageBox.critical(self, "Burst Error", err)
        self.burst_thread.quit()

    def save_full_dataset(self):
        try:
            # Gather Metadata
            meta = ExperimentMetadata(
                exposure_s = self.controls.spin_exp.value() / 1000.0,
                gain_db = self.controls.spin_gain.value(),
                wavelength_nm = self.controls.spin_lambda.value(),
                slit_separation_mm = self.controls.spin_slit.value(),
                distance_m = self.controls.spin_dist.value()
            )
            
            if not hasattr(self, 'last_fit_result') or not self.last_fit_result:
                QMessageBox.warning(self, "No Data", "No fit result to save.")
                return

            res = ExperimentResult(
                visibility = self.last_fit_result.visibility,
                sigma_microns = self.last_fit_result.sigma_microns,
                lineout_y = self.current_lineout if hasattr(self, 'current_lineout') else [],
                fit_y = self.last_fit_result.fitted_curve if self.last_fit_result.fitted_curve is not None else [],
                is_saturated = (self.controls.lbl_sat.text() != "OK")
            )

            dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
            if dir_path:
                saved_path = DataManager.save_dataset(dir_path, "SRI_Data", self.current_raw_image, meta, res)
                self.logger.info(f"Saved to {saved_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def save_mat_file(self):
        try:
            if not hasattr(self, 'current_raw_image'): return
            
            meta = ExperimentMetadata(
                exposure_s = self.controls.spin_exp.value() / 1000.0,
                gain_db = self.controls.spin_gain.value(),
                wavelength_nm = self.controls.spin_lambda.value(),
                slit_separation_mm = self.controls.spin_slit.value(),
                distance_m = self.controls.spin_dist.value()
            )
            # Create a simple result if fit hasn't run yet
            res_args = {}
            if hasattr(self, 'last_fit_result') and self.last_fit_result:
                res_args['visibility'] = self.last_fit_result.visibility
                res_args['sigma_microns'] = self.last_fit_result.sigma_microns
                res_args['fit_y'] = self.last_fit_result.fitted_curve
            
            res = ExperimentResult(**res_args)
            if hasattr(self, 'current_lineout'):
                res.lineout_y = self.current_lineout

            dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
            if dir_path:
                saved_path = DataManager.save_matlab(dir_path, "SRI_Matlab", self.current_raw_image, meta, res)
                self.logger.info(f"Saved MATLAB file to {saved_path}")
                QMessageBox.information(self, "Saved", f"Saved .mat file:\n{os.path.basename(saved_path)}")
        except Exception as e:
            QMessageBox.critical(self, "MATLAB Save Failed", str(e))

    # --- Helper Logic ---

    def ensure_roi_bounds(self, h, w):
        r_min, r_max = self.live_widget.get_roi_rows()
        if r_min > h or r_max < 0: self.live_widget.set_roi_rows(h*0.25, h*0.75)
        
        c_min, c_max = self.live_widget.get_roi_width()
        if c_min > w or c_max < 0: self.live_widget.set_roi_width(w*0.25, w*0.75)

    def run_autocenter(self, lineout, w):
        peak_idx = np.argmax(lineout)
        if lineout[peak_idx] > 200 and 5 < peak_idx < (w-5):
            c_min, c_max = self.live_widget.get_roi_width()
            width = c_max - c_min
            new_min = max(0, peak_idx - width/2)
            new_max = min(w, peak_idx + width/2)
            if abs(new_min - c_min) > 2:
                self.live_widget.set_roi_width(new_min, new_max)

    def reset_rois(self):
        if not hasattr(self, 'current_raw_image'): return
        img = self.current_raw_image
        if self.controls.chk_transpose.isChecked():
            h, w = img.shape[1], img.shape[0]
        else:
            h, w = img.shape[0], img.shape[1]
        self.live_widget.set_roi_rows(h*0.25, h*0.75)
        self.live_widget.set_roi_width(w*0.25, w*0.75)

    def toggle_live_view(self, checked):
        if checked:
            if not self.cam_thread.isRunning():
                self.cam_thread.start()
            self.controls.btn_live.setText("Stop Live View")
            self.controls.btn_live.setStyleSheet("background-color: red; color: white; font-weight: bold;")
            self.start_worker_signal.emit()
        else:
            self.controls.btn_live.setText("Start Live View")
            self.controls.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold;")
            self.cam_worker.stop_acquire()

    def update_physics(self):
        self.fitter.wavelength = self.controls.spin_lambda.value() * 1e-9
        self.fitter.slit_sep = self.controls.spin_slit.value() * 1e-3
        self.fitter.distance = self.controls.spin_dist.value()

    def update_exposure(self, val): self.driver.exposure = val / 1000.0
    def update_gain(self, val): self.driver.gain = val

    def apply_config(self):
        c = self.cfg
        cp = self.controls
        
        cp.spin_exp.setValue(c['camera']['exposure_ms'])
        cp.spin_gain.setValue(c['camera']['gain_db'])
        cp.chk_transpose.setChecked(c['camera']['transpose'])
        
        cp.spin_lambda.setValue(c['physics']['wavelength_nm'])
        cp.spin_slit.setValue(c['physics']['slit_separation_mm'])
        cp.spin_dist.setValue(c['physics']['distance_m'])
        
        self.live_widget.set_roi_rows(c['roi']['rows_min'], c['roi']['rows_max'])
        self.live_widget.set_roi_width(c['roi']['fit_width_min'], c['roi']['fit_width_max'])
        cp.chk_autocenter.setChecked(c['roi']['auto_center'])
        
        self.update_physics()

    def closeEvent(self, event):
        # Save Config
        state = {
            "camera": { "exposure_ms": self.controls.spin_exp.value(), 
                        "gain_db": self.controls.spin_gain.value(),
                        "transpose": self.controls.chk_transpose.isChecked() },
            "physics": { "wavelength_nm": self.controls.spin_lambda.value(), 
                         "slit_separation_mm": self.controls.spin_slit.value(),
                         "distance_m": self.controls.spin_dist.value() },
            "roi": { "rows_min": self.live_widget.get_roi_rows()[0], 
                     "rows_max": self.live_widget.get_roi_rows()[1],
                     "fit_width_min": self.live_widget.get_roi_width()[0], 
                     "fit_width_max": self.live_widget.get_roi_width()[1],
                     "auto_center": self.controls.chk_autocenter.isChecked() }
        }
        self.config_manager.save(state)
        
        # Shutdown
        self.cam_worker.stop_acquire()
        self.cam_thread.quit()
        self.cam_thread.wait()
        
        self.an_thread.quit()
        self.an_thread.wait()
        
        if hasattr(self, 'driver'): 
            self.driver.close()
        event.accept()
