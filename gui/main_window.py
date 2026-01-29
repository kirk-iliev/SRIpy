import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, 
                             QGroupBox, QGridLayout, QMessageBox, QCheckBox, 
                             QTabWidget, QProgressBar)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject
import pyqtgraph as pg
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from hardware.manta_driver import MantaDriver
from analysis.fitter import InterferenceFitter

try:
    from core.data_model import BurstResult, DataManager, ExperimentMetadata, ExperimentResult
except ImportError:
    from core.data_model import BurstResult, DataManager, ExperimentMetadata, ExperimentResult

class AnalysisWorker(QObject):
    result_ready = pyqtSignal(object, object)

    def __init__(self, fitter):
        super().__init__()
        self.fitter = fitter

    def process_fit(self, y_data, x_data):
        try:
            fit_result = self.fitter.fit(y_data)
            self.result_ready.emit(fit_result, x_data)
        except Exception as e:
            from analysis.fitter import FitResult
            failed_res = FitResult(success=False, message=str(e))
            self.result_ready.emit(failed_res, x_data)

class BurstWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object) 
    error = pyqtSignal(str)

    def __init__(self, driver, fitter, n_frames, roi_slice, roi_x_map):
        super().__init__()
        self.driver = driver
        self.fitter = fitter
        self.n_frames = n_frames
        self.roi_slice = roi_slice
        self.roi_x_map = roi_x_map 

    def run_burst(self):
        try:
            vis_list = []
            sigma_list = []
            timestamps = []
            lineout_list = [] 
            
            self.driver.start_stream()
            
            for i in range(self.n_frames):
                img = self.driver.acquire_frame(timeout=2.0)
                if img is None: continue

                img = img.squeeze().astype(np.float32)
                
                if self.roi_slice:
                    lineout = self.fitter.get_lineout(img, self.roi_slice)
                else:
                    lineout = self.fitter.get_lineout(img)
                
                if self.roi_x_map is not None:
                    x_min, x_max = self.roi_x_map
                    x_min = max(0, x_min)
                    x_max = min(len(lineout), x_max)
                    y_fit_data = lineout[x_min:x_max]
                else:
                    y_fit_data = lineout

                y_fit_data = np.nan_to_num(y_fit_data)
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
            pass

class InterferometerGUI(QMainWindow):
    request_fit = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRIpy Interferometer Monitor")
        self.resize(1300, 950) 

        try:
            self.driver = MantaDriver() 
            self.driver.connect()
            print("Hardware initialized successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Could not connect to camera:\n{e}")
            sys.exit(1)

        self.fitter = InterferenceFitter()
        self.background_frame = None 
        
        self.history_len = 100
        self.history_sigma = np.zeros(self.history_len)
        self.history_vis = np.zeros(self.history_len)
        
        self.user_is_interacting = False
        
        self.thread = QThread()
        self.worker = AnalysisWorker(self.fitter)
        self.worker.moveToThread(self.thread)
        self.request_fit.connect(self.worker.process_fit)
        self.worker.result_ready.connect(self.handle_fit_result)
        self.thread.start()
        self.worker_is_busy = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        plot_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        plot_layout.addWidget(self.tabs)
        
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.image_container = pg.GraphicsLayoutWidget()
        self.image_plot = self.image_container.addPlot(title="Camera Frame")
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)
        self.roi_rows = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Horizontal, brush=(0, 50, 255, 50))
        self.roi_rows.setRegion([400, 800]) 
        self.image_plot.addItem(self.roi_rows)
        
        self.lineout_plot = pg.PlotWidget(title="Interference Profile")
        self.lineout_plot.showGrid(x=True, y=True)
        self.curve_raw = self.lineout_plot.plot(pen=pg.mkPen('w', width=2), name="Raw Data")
        self.curve_fit = self.lineout_plot.plot(pen=pg.mkPen('r', width=3, style=Qt.PenStyle.DashLine), name="Fit")
        self.roi_fit_width = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical, brush=(0, 255, 0, 30))
        self.roi_fit_width.setRegion([800, 1200]) 
        self.lineout_plot.addItem(self.roi_fit_width)
        
        self.roi_fit_width.sigRegionChanged.connect(self.roi_drag_start)
        self.roi_fit_width.sigRegionChangeFinished.connect(self.roi_drag_end)

        tab1_layout.addWidget(self.image_container, stretch=3)
        tab1_layout.addWidget(self.lineout_plot, stretch=2)
        self.tabs.addTab(tab1, "Live Monitor")

        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        self.history_plot = pg.PlotWidget(title="Beam Size History")
        self.history_plot.setLabel('left', 'Sigma (um)')
        self.history_plot.setLabel('bottom', 'Frame History')
        self.history_plot.showGrid(x=True, y=True)
        self.curve_history = self.history_plot.plot(pen=pg.mkPen('c', width=2), symbol='o', symbolSize=5)
        tab2_layout.addWidget(self.history_plot)
        self.tabs.addTab(tab2, "Stability History")
        
        controls_layout = QVBoxLayout()
        
        stats_group = QGroupBox("Analysis Results")
        stats_layout = QGridLayout()
        self.lbl_vis = QLabel("0.000")
        self.lbl_vis.setStyleSheet("font-size: 28px; font-weight: bold; color: yellow;")
        stats_layout.addWidget(QLabel("Visibility:"), 0, 0)
        stats_layout.addWidget(self.lbl_vis, 0, 1)
        self.lbl_sigma = QLabel("0.0 um")
        self.lbl_sigma.setStyleSheet("font-size: 28px; font-weight: bold; color: cyan;")
        stats_layout.addWidget(QLabel("Beam Sigma:"), 1, 0)
        stats_layout.addWidget(self.lbl_sigma, 1, 1)
        self.lbl_sat = QLabel("OK")
        self.lbl_sat.setStyleSheet("color: green; font-weight: bold;")
        stats_layout.addWidget(QLabel("Sensor Status:"), 2, 0)
        stats_layout.addWidget(self.lbl_sat, 2, 1)
        stats_group.setLayout(stats_layout)
        controls_layout.addWidget(stats_group)

        roi_group = QGroupBox("ROI Controls")
        roi_layout = QVBoxLayout()
        self.chk_autocenter = QCheckBox("Auto-Center ROI")
        self.chk_autocenter.setChecked(True) 
        roi_layout.addWidget(self.chk_autocenter)
        roi_group.setLayout(roi_layout)
        controls_layout.addWidget(roi_group)

        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout()
        self.btn_bg = QPushButton("Acquire Background")
        self.btn_bg.setStyleSheet("background-color: #444; color: white; height: 30px;")
        self.btn_bg.clicked.connect(self.acquire_background)
        cal_layout.addWidget(self.btn_bg)
        self.chk_bg = QCheckBox("Subtract Background")
        self.chk_bg.setEnabled(False) 
        cal_layout.addWidget(self.chk_bg)
        cal_group.setLayout(cal_layout)
        controls_layout.addWidget(cal_group)

        cam_group = QGroupBox("Camera Settings")
        cam_layout = QGridLayout()
        cam_layout.addWidget(QLabel("Exposure (ms):"), 0, 0)
        self.spin_exp = QDoubleSpinBox()
        self.spin_exp.setRange(0.05, 1000)
        self.spin_exp.setValue(5.0) 
        self.spin_exp.setSingleStep(0.5)
        self.spin_exp.valueChanged.connect(self.update_exposure)
        cam_layout.addWidget(self.spin_exp, 0, 1)
        cam_layout.addWidget(QLabel("Gain (dB):"), 1, 0)
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0, 40)
        self.spin_gain.setValue(0)
        self.spin_gain.valueChanged.connect(self.update_gain)
        cam_layout.addWidget(self.spin_gain, 1, 1)
        cam_group.setLayout(cam_layout)
        controls_layout.addWidget(cam_group)

        phys_group = QGroupBox("Experiment Physics")
        phys_layout = QGridLayout()
        phys_layout.addWidget(QLabel("Wavelength (nm):"), 0, 0)
        self.spin_lambda = QDoubleSpinBox()
        self.spin_lambda.setRange(200, 1500)
        self.spin_lambda.setValue(550.0)
        self.spin_lambda.valueChanged.connect(self.update_physics)
        phys_layout.addWidget(self.spin_lambda, 0, 1)
        phys_layout.addWidget(QLabel("Slit Sep (mm):"), 1, 0)
        self.spin_slit = QDoubleSpinBox()
        self.spin_slit.setRange(0.1, 1000.0)
        self.spin_slit.setValue(50.0) 
        phys_layout.addWidget(self.spin_slit, 1, 1)
        phys_layout.addWidget(QLabel("Distance (m):"), 2, 0)
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.1, 100.0)
        self.spin_dist.setValue(16.5)
        phys_layout.addWidget(self.spin_dist, 2, 1)
        phys_group.setLayout(phys_layout)
        controls_layout.addWidget(phys_group)
        
        self.btn_live = QPushButton("Start Live View")
        self.btn_live.setCheckable(True)
        self.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px;")
        self.btn_live.clicked.connect(self.toggle_live_view)
        controls_layout.addWidget(self.btn_live)
        
        self.btn_burst = QPushButton("Burst Acquire (50)")
        self.btn_burst.setStyleSheet("background-color: #d95f02; color: white; font-weight: bold; height: 40px;")
        self.btn_burst.clicked.connect(self.start_burst_acquisition)
        controls_layout.addWidget(self.btn_burst)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 50)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        self.btn_save_data = QPushButton("Save Dataset (Folder)")
        self.btn_save_data.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; height: 30px;")
        self.btn_save_data.clicked.connect(self.save_full_dataset)
        controls_layout.addWidget(self.btn_save_data)

        self.btn_save_mat = QPushButton("Save for MATLAB (.mat)")
        self.btn_save_mat.setStyleSheet("background-color: #7570b3; color: white; font-weight: bold; height: 30px;")
        self.btn_save_mat.clicked.connect(self.save_mat_file)
        controls_layout.addWidget(self.btn_save_mat)
        
        controls_layout.addStretch()
        main_layout.addLayout(plot_layout, stretch=4)
        main_layout.addLayout(controls_layout, stretch=1)

        self.update_exposure(self.spin_exp.value())
        self.update_gain(self.spin_gain.value())
        self.update_physics() 

    def roi_drag_start(self): self.user_is_interacting = True
    def roi_drag_end(self): self.user_is_interacting = False
    
    def update_physics(self):
        wl = self.spin_lambda.value() * 1e-9  
        slit = self.spin_slit.value() * 1e-3  
        dist = self.spin_dist.value()        
        self.fitter.wavelength = wl
        self.fitter.slit_sep = slit
        self.fitter.distance = dist

    def acquire_background(self):
        try:
            was_running = self.timer.isActive()
            if was_running: self.timer.stop()
            raw = self.driver.acquire_frame()
            if raw is not None:
                raw = raw.squeeze()
                self.background_frame = raw.astype(np.float32)
                self.chk_bg.setEnabled(True)
                self.chk_bg.setChecked(True)
                self.chk_bg.setText("Subtract Background (Active)")
                print("Background acquired.")
            
            if was_running: self.timer.start(50)
            
        except Exception as e:
            print(f"Failed to acquire background: {e}")

    def toggle_live_view(self, checked):
        if checked:
            self.btn_live.setText("Stop Live View")
            self.btn_live.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 14px;")
            if hasattr(self, 'driver'):
                self.driver.start_stream()
            self.timer.start(50) 
        else:
            self.btn_live.setText("Start Live View")
            self.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px;")
            self.timer.stop()
            if hasattr(self, 'driver'):
                self.driver.stop_stream()

    def update_frame(self):
        try:
            img_int = self.driver.acquire_frame()
            if img_int is None: 
                return # Skip this cycle if acquisition timed out

            img = img_int.squeeze().astype(np.float32)
            self.current_raw_image = img 
            
            max_val = np.max(img)
            if max_val >= 4090:
                self.lbl_sat.setText("SATURATED!")
                self.lbl_sat.setStyleSheet("color: red; font-weight: bold; font-size: 16px; background-color: yellow;")
            else:
                self.lbl_sat.setText("OK")
                self.lbl_sat.setStyleSheet("color: green; font-weight: bold;")

            if self.chk_bg.isChecked() and self.background_frame is not None:
                img = img - self.background_frame
                img[img < 0] = 0
            
            self.image_item.setImage(img.T, autoLevels=False, levels=(0, 4095))
            
            y_min, y_max = self.roi_rows.getRegion()
            y_min, y_max = int(y_min), int(y_max)
            h, w = img.shape
            y_min = max(0, y_min)
            y_max = min(h, y_max)
            
            if y_max > y_min:
                roi_slice = slice(y_min, y_max)
                lineout = self.fitter.get_lineout(img, roi_slice=roi_slice)
                self.current_lineout = lineout
            else:
                lineout = np.zeros(w)
                self.current_lineout = lineout

            self.curve_raw.setData(lineout)

            
            # Only recalculate if auto-center is ON and user isn't dragging ROI
            if self.chk_autocenter.isChecked() and not self.user_is_interacting:
                peak_val = np.max(lineout)
                
                # This ignores dark noise/hot pixels when lens is covered.
                if peak_val > 200:  
                    peak_x = np.argmax(lineout)
                    
                    # We check if peak_x is within safe bounds (e.g. 5 pixels from edge)
                    is_edge_noise = (peak_x <= 5) or (peak_x >= w - 5)
                    
                    if not is_edge_noise:
                        current_min, current_max = self.roi_fit_width.getRegion()
                        width = current_max - current_min
                        
                        new_min = peak_x - (width / 2)
                        new_max = peak_x + (width / 2)
                        
                        new_min = max(0, new_min)
                        new_max = min(w, new_max)
                        
                        # This prevents "shivering" on static beams
                        if abs(new_min - current_min) > 2:
                            self.roi_fit_width.setRegion([new_min, new_max])

            if not self.worker_is_busy:
                x_min, x_max = self.roi_fit_width.getRegion()
                x_min, x_max = int(x_min), int(x_max)
                x_min = max(0, x_min)
                x_max = min(w, x_max)

                if x_max > x_min:
                    x_data = np.arange(x_min, x_max)
                    y_data = lineout[x_min:x_max]
                    self.worker_is_busy = True
                    self.request_fit.emit(y_data, x_data)

        except Exception as e:
            print(f"Error in update loop: {e}")
            self.timer.stop()
            self.btn_live.setChecked(False)
            self.toggle_live_view(False)

    def handle_fit_result(self, fit_result, x_data):
        self.worker_is_busy = False
        self.last_fit_result = fit_result
        
        if fit_result.success:
            self.curve_fit.setData(x_data, fit_result.fitted_curve)
            self.lbl_vis.setText(f"{fit_result.visibility:.3f}")
            self.lbl_sigma.setText(f"{fit_result.sigma_microns:.1f} um")
            self.history_sigma = np.roll(self.history_sigma, -1)
            self.history_sigma[-1] = fit_result.sigma_microns
            self.curve_history.setData(self.history_sigma)
        else:
            self.curve_fit.setData([], [])
            self.lbl_vis.setText("---")

    def start_burst_acquisition(self):
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()
            self.btn_live.setText("Start Live View")
            self.btn_live.setChecked(False)
            self.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px;")
        
        y_min, y_max = self.roi_rows.getRegion()
        roi_slice = slice(int(y_min), int(y_max))
        
        x_min, x_max = self.roi_fit_width.getRegion()
        roi_x_map = (int(x_min), int(x_max))
        
        self.burst_thread = QThread()
        self.burst_worker = BurstWorker(self.driver, self.fitter, 50, roi_slice, roi_x_map)
        self.burst_worker.moveToThread(self.burst_thread)
        
        self.burst_thread.started.connect(self.burst_worker.run_burst)
        self.burst_worker.progress.connect(self.progress_bar.setValue)
        self.burst_worker.finished.connect(self.handle_burst_finished)
        self.burst_worker.error.connect(self.handle_burst_error)
        self.burst_worker.finished.connect(self.burst_thread.quit)
        self.burst_worker.finished.connect(self.burst_worker.deleteLater)
        self.burst_thread.finished.connect(self.burst_thread.deleteLater)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_burst.setEnabled(False)
        self.btn_live.setEnabled(False)
        
        self.burst_thread.start()

    def handle_burst_finished(self, burst_result):
        self.btn_burst.setEnabled(True)
        self.btn_live.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        msg = (f"Burst Complete (50 frames)\n\n"
               f"Mean Sigma: {burst_result.mean_sigma:.2f} um\n"
               f"Std Dev:    {burst_result.std_sigma:.2f} um\n\n"
               f"Mean Vis:   {burst_result.mean_visibility:.3f}")
        
        QMessageBox.information(self, "Burst Results", msg)
        self.last_burst_result = burst_result
        self.history_sigma = np.roll(self.history_sigma, -1)
        self.history_sigma[-1] = burst_result.mean_sigma
        self.curve_history.setData(self.history_sigma)

    def handle_burst_error(self, err_msg):
        self.btn_burst.setEnabled(True)
        self.btn_live.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Burst Error", err_msg)
        self.burst_thread.quit()

    def update_exposure(self, val):
        if hasattr(self, 'driver'):
            self.driver.exposure = val / 1000.0

    def update_gain(self, val):
        if hasattr(self, 'driver'):
            self.driver.gain = val

    def save_full_dataset(self):
        was_running = self.timer.isActive()
        if was_running: 
            self.toggle_live_view(False)
            self.btn_live.setChecked(False)

        try:
            try:
                from core.data_model import DataManager, ExperimentMetadata, ExperimentResult
            except ImportError:
                from core.data_model import DataManager, ExperimentMetadata, ExperimentResult
            
            meta = ExperimentMetadata(
                exposure_s = self.spin_exp.value() / 1000.0,
                gain_db = self.spin_gain.value(),
                wavelength_nm = self.spin_lambda.value(),
                slit_separation_mm = self.spin_slit.value(),
                distance_m = self.spin_dist.value()
            )
            
            if not hasattr(self, 'last_fit_result') or not self.last_fit_result:
                QMessageBox.warning(self, "No Data", "No analysis data available to save.")
                return

            res = ExperimentResult(
                visibility = self.last_fit_result.visibility,
                sigma_microns = self.last_fit_result.sigma_microns,
                lineout_y = self.current_lineout if hasattr(self, 'current_lineout') else [],
                fit_y = self.last_fit_result.fitted_curve if self.last_fit_result.fitted_curve is not None else [],
                is_saturated = (self.lbl_sat.text() != "OK")
            )
            
            if not hasattr(self, 'current_raw_image'):
                QMessageBox.warning(self, "No Data", "No image data available.")
                return

            from PyQt6.QtWidgets import QFileDialog
            dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
            if dir_path:
                saved_path = DataManager.save_dataset(dir_path, "SRI_Data", self.current_raw_image, meta, res)
                
                if hasattr(self, 'last_burst_result'):
                    DataManager.save_burst(dir_path, "SRI_Data", self.last_burst_result, meta)
                    
                print(f"Saved to {saved_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")
            
        finally:
            if was_running: 
                self.toggle_live_view(True)
                self.btn_live.setChecked(True)

    def save_mat_file(self):
        was_running = self.timer.isActive()
        if was_running: 
            self.toggle_live_view(False)
            self.btn_live.setChecked(False)

        try:
            if not hasattr(self, 'current_raw_image') or not hasattr(self, 'last_fit_result') or not self.last_fit_result:
                QMessageBox.warning(self, "No Data", "No valid data to save yet.")
                return

            try:
                from core.data_model import DataManager, ExperimentMetadata, ExperimentResult
            except ImportError:
                from core.data_model import DataManager, ExperimentMetadata, ExperimentResult
            
            meta = ExperimentMetadata(
                exposure_s = self.spin_exp.value() / 1000.0,
                gain_db = self.spin_gain.value(),
                wavelength_nm = self.spin_lambda.value(),
                slit_separation_mm = self.spin_slit.value(),
                distance_m = self.spin_dist.value()
            )
            
            res = ExperimentResult(
                visibility = self.last_fit_result.visibility,
                sigma_microns = self.last_fit_result.sigma_microns,
                lineout_y = self.current_lineout if hasattr(self, 'current_lineout') else [],
                fit_y = self.last_fit_result.fitted_curve if self.last_fit_result.fitted_curve is not None else [],
                is_saturated = (self.lbl_sat.text() != "OK")
            )

            from PyQt6.QtWidgets import QFileDialog
            dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
            
            if dir_path:
                saved_path = DataManager.save_matlab(dir_path, "SRI_Matlab", self.current_raw_image, meta, res)
                print(f"Saved MATLAB file to {saved_path}")
                QMessageBox.information(self, "Saved", f"Saved .mat file:\n{os.path.basename(saved_path)}")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save .mat:\n{e}")
            
        finally:
            if was_running: 
                self.toggle_live_view(True)
                self.btn_live.setChecked(True)

    def save_snapshot(self):
        import pyqtgraph.exporters
        exporter = pyqtgraph.exporters.ImageExporter(self.lineout_plot.plotItem)
        exporter.export("snapshot_lineout.png")
        print("Saved snapshot_lineout.png")

    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'driver'):
            self.driver.close()
        self.thread.quit()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = InterferometerGUI()
    window.show()
    sys.exit(app.exec())