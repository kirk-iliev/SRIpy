import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, 
                             QGroupBox, QGridLayout, QMessageBox)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
import os

# 1. Get the path to the current file (gui/main_window.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (SRIpy root)
project_root = os.path.dirname(current_dir)

# 3. Add root to Python's search path
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORT OUR CUSTOM MODULES ---
from hardware.manta_driver import MantaDriver
from analysis.fitter import InterferenceFitter

class InterferometerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRIpy Interferometer Monitor")
        self.resize(1200, 800)

        # --- 1. INITIALIZE BACKEND ---
        try:
            # We use a specific ID if you know it, otherwise None finds first cam
            # self.driver = MantaDriver("DEV_000F31501803") 
            self.driver = MantaDriver() 
            self.driver.connect()
            print("Hardware initialized successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Could not connect to camera:\n{e}")
            sys.exit(1) # Crash gracefully if no camera

        self.fitter = InterferenceFitter()
        
        # Acquisition timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Build UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Column: Visualization
        plot_layout = QVBoxLayout()
        self.lineout_plot = pg.PlotWidget(title="Interference Profile")
        self.lineout_plot.setLabel('left', 'Intensity (a.u.)')
        self.lineout_plot.setLabel('bottom', 'Pixel')
        self.lineout_plot.showGrid(x=True, y=True)
        self.curve_raw = self.lineout_plot.plot(pen=pg.mkPen('w', width=2), name="Raw Data")
        self.curve_fit = self.lineout_plot.plot(pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine), name="Fit")
        
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        
        plot_layout.addWidget(self.lineout_plot, stretch=1)
        plot_layout.addWidget(self.image_view, stretch=1)
        
        # Right Column: Controls
        controls_layout = QVBoxLayout()
        
        # Stats Group
        stats_group = QGroupBox("Analysis Results")
        stats_layout = QGridLayout()
        self.lbl_vis = QLabel("0.000")
        self.lbl_vis.setStyleSheet("font-size: 24px; font-weight: bold; color: yellow;")
        stats_layout.addWidget(QLabel("Visibility:"), 0, 0)
        stats_layout.addWidget(self.lbl_vis, 0, 1)
        
        self.lbl_sigma = QLabel("0.0 um")
        self.lbl_sigma.setStyleSheet("font-size: 24px; font-weight: bold; color: cyan;")
        stats_layout.addWidget(QLabel("Beam Sigma:"), 1, 0)
        stats_layout.addWidget(self.lbl_sigma, 1, 1)
        stats_group.setLayout(stats_layout)
        controls_layout.addWidget(stats_group)

        # Camera Settings
        cam_group = QGroupBox("Camera Settings")
        cam_layout = QGridLayout()
        
        cam_layout.addWidget(QLabel("Exposure (ms):"), 0, 0)
        self.spin_exp = QDoubleSpinBox()
        self.spin_exp.setRange(0.05, 1000)
        self.spin_exp.setValue(5.0) 
        self.spin_exp.setSingleStep(0.5)
        self.spin_exp.valueChanged.connect(self.update_exposure) # Connect signal
        cam_layout.addWidget(self.spin_exp, 0, 1)
        
        cam_layout.addWidget(QLabel("Gain (dB):"), 1, 0)
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0, 40)
        self.spin_gain.setValue(0)
        self.spin_gain.valueChanged.connect(self.update_gain) # Connect signal
        cam_layout.addWidget(self.spin_gain, 1, 1)
        
        cam_group.setLayout(cam_layout)
        controls_layout.addWidget(cam_group)
        
        # Buttons
        self.btn_live = QPushButton("Start Live View")
        self.btn_live.setCheckable(True)
        self.btn_live.setMinimumHeight(50)
        self.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px;")
        self.btn_live.clicked.connect(self.toggle_live_view)
        controls_layout.addWidget(self.btn_live)
        
        self.btn_snap = QPushButton("Save Snapshot (PNG)")
        self.btn_snap.clicked.connect(self.save_snapshot)
        self.btn_snap.setMinimumHeight(40)
        controls_layout.addWidget(self.btn_snap)
        
        controls_layout.addStretch()
        main_layout.addLayout(plot_layout, stretch=3)
        main_layout.addLayout(controls_layout, stretch=1)

        # Apply Initial Settings
        self.update_exposure(self.spin_exp.value())
        self.update_gain(self.spin_gain.value())

    # --- 3. LOGIC METHODS ---

    def toggle_live_view(self, checked):
        if checked:
            self.btn_live.setText("Stop Live View")
            self.btn_live.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 14px;")
            # Start the timer loop (e.g., every 100ms)
            self.timer.start(100) 
        else:
            self.btn_live.setText("Start Live View")
            self.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px;")
            self.timer.stop()

    def update_frame(self):
        """The Main Loop: Acquire -> Fit -> Plot"""
        try:
            # 1. Acquire
            img = self.driver.acquire_frame()
            
            # squeeze extra dim: (1216, 1936, 1) -> (1216, 1936)
            img = img.squeeze() 
            
            # 2. Update Image Plot
            # Transpose (.T) aligns the image so 'Left-Right' on the screen matches 'Left-Right' on the plot
            # autoLevels=True ensures it adjusts contrast automatically
            self.image_view.setImage(img.T, autoRange=False, autoLevels=True)
            
            # 3. Analyze
            lineout = self.fitter.get_lineout(img)
            fit_result = self.fitter.fit(lineout)
            
            # 4. Update Lineout Plot
            self.curve_raw.setData(lineout)
            
            if fit_result:
                # Plot the red fit line
                self.curve_fit.setData(fit_result['fitted_curve'])
                
                # Update Labels
                vis = fit_result['visibility']
                sigma = fit_result['sigma'] * 1e6 # Convert to microns
                
                self.lbl_vis.setText(f"{vis:.3f}")
                self.lbl_sigma.setText(f"{sigma:.1f} um")
            else:
                # Clear fit line if failed
                self.curve_fit.setData([])
                self.lbl_vis.setText("---")

        except Exception as e:
            print(f"Error in update loop: {e}")
            self.timer.stop()
            self.btn_live.setChecked(False)
            self.toggle_live_view(False)

    def update_exposure(self, val):
        # Convert ms to seconds
        try:
            self.driver.set_exposure(val / 1000.0)
        except Exception as e:
            print(f"Error setting exposure: {e}")

    def update_gain(self, val):
        try:
            self.driver.set_gain(val)
        except Exception as e:
            print(f"Error setting gain: {e}")

    def save_snapshot(self):
        # Quick and dirty save functionality for now
        import pyqtgraph.exporters
        exporter = pyqtgraph.exporters.ImageExporter(self.lineout_plot.plotItem)
        exporter.export("snapshot_lineout.png")
        print("Saved snapshot_lineout.png")

    def closeEvent(self, event):
        # Cleanup when closing window
        self.timer.stop()
        if hasattr(self, 'driver'):
            self.driver.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = InterferometerGUI()
    window.show()
    sys.exit(app.exec())