from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QGridLayout, 
                             QLabel, QDoubleSpinBox, QCheckBox, QPushButton, QProgressBar)
from PyQt6.QtCore import pyqtSignal

class ControlPanelWidget(QWidget):
    # Define signals for all user interactions
    exposure_changed = pyqtSignal(float)
    gain_changed = pyqtSignal(float)
    physics_changed = pyqtSignal() # Generalized signal for lambda/slit/dist
    acquire_bg_clicked = pyqtSignal()
    toggle_live_clicked = pyqtSignal(bool)
    burst_clicked = pyqtSignal()
    save_data_clicked = pyqtSignal()
    save_mat_clicked = pyqtSignal()
    reset_roi_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # --- Analysis Results ---
        stats_group = QGroupBox("Analysis Results")
        sl = QGridLayout()
        self.lbl_vis = QLabel("0.000")
        self.lbl_vis.setStyleSheet("font-size: 28px; font-weight: bold; color: yellow;")
        self.lbl_sigma = QLabel("0.0 um")
        self.lbl_sigma.setStyleSheet("font-size: 28px; font-weight: bold; color: cyan;")
        self.lbl_sat = QLabel("OK")
        self.lbl_sat.setStyleSheet("color: green; font-weight: bold;")
        
        sl.addWidget(QLabel("Visibility:"), 0, 0); sl.addWidget(self.lbl_vis, 0, 1)
        sl.addWidget(QLabel("Beam Sigma:"), 1, 0); sl.addWidget(self.lbl_sigma, 1, 1)
        sl.addWidget(QLabel("Sensor Status:"), 2, 0); sl.addWidget(self.lbl_sat, 2, 1)
        stats_group.setLayout(sl)
        layout.addWidget(stats_group)

        # --- ROI Controls ---
        roi_group = QGroupBox("ROI Controls")
        rl = QVBoxLayout()
        self.chk_autocenter = QCheckBox("Auto-Center ROI")
        self.chk_autocenter.setChecked(True)
        rl.addWidget(self.chk_autocenter)
        
        self.btn_reset_roi = QPushButton("Reset / Find ROIs")
        self.btn_reset_roi.clicked.connect(lambda: self.reset_roi_clicked.emit())
        rl.addWidget(self.btn_reset_roi)
        roi_group.setLayout(rl)
        layout.addWidget(roi_group)

        # --- Calibration ---
        cal_group = QGroupBox("Calibration")
        cl = QVBoxLayout()
        self.btn_bg = QPushButton("Acquire Background")
        self.btn_bg.clicked.connect(lambda: self.acquire_bg_clicked.emit())
        cl.addWidget(self.btn_bg)
        self.chk_bg = QCheckBox("Subtract Background")
        self.chk_bg.setEnabled(False)
        cl.addWidget(self.chk_bg)
        cal_group.setLayout(cl)
        layout.addWidget(cal_group)

        # --- Camera Settings ---
        cam_group = QGroupBox("Camera Settings")
        cgl = QGridLayout()
        self.chk_transpose = QCheckBox("Transpose Image")
        cgl.addWidget(self.chk_transpose, 0, 2)
        
        cgl.addWidget(QLabel("Exp (ms):"), 0, 0)
        self.spin_exp = QDoubleSpinBox()
        self.spin_exp.setRange(0.05, 1000); self.spin_exp.setValue(5.0); self.spin_exp.setSingleStep(0.5)
        # Forward the numeric value to the exposure_changed signal
        self.spin_exp.valueChanged.connect(self.exposure_changed.emit)
        cgl.addWidget(self.spin_exp, 0, 1)
        
        cgl.addWidget(QLabel("Gain (dB):"), 1, 0)
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0, 40)
        # Forward the numeric value to the gain_changed signal
        self.spin_gain.valueChanged.connect(self.gain_changed.emit)
        cgl.addWidget(self.spin_gain, 1, 1)
        cam_group.setLayout(cgl)
        layout.addWidget(cam_group)

        # --- Physics ---
        phys_group = QGroupBox("Experiment Physics")
        pl = QGridLayout()
        self.spin_lambda = QDoubleSpinBox(); self.spin_lambda.setRange(200, 1500); self.spin_lambda.setValue(550.0)
        self.spin_slit = QDoubleSpinBox(); self.spin_slit.setRange(0.1, 1000); self.spin_slit.setValue(50.0)
        self.spin_dist = QDoubleSpinBox(); self.spin_dist.setRange(0.1, 100); self.spin_dist.setValue(16.5)
        
        for s in [self.spin_lambda, self.spin_slit, self.spin_dist]:
            # Drop the numeric argument and emit a generic physics_changed signal
            s.valueChanged.connect(lambda _val, _self=self: _self.physics_changed.emit())

        pl.addWidget(QLabel("Lambda (nm):"), 0, 0); pl.addWidget(self.spin_lambda, 0, 1)
        pl.addWidget(QLabel("Slit Sep (mm):"), 1, 0); pl.addWidget(self.spin_slit, 1, 1)
        pl.addWidget(QLabel("Distance (m):"), 2, 0); pl.addWidget(self.spin_dist, 2, 1)
        phys_group.setLayout(pl)
        layout.addWidget(phys_group)

        # --- Actions ---
        self.btn_live = QPushButton("Start Live View")
        self.btn_live.setCheckable(True)
        self.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 14px;")
        # Forward the toggle boolean to the toggle_live_clicked signal
        self.btn_live.toggled.connect(self.toggle_live_clicked.emit)
        layout.addWidget(self.btn_live)

        self.btn_burst = QPushButton("Burst Acquire (50)")
        self.btn_burst.setStyleSheet("background-color: #d95f02; color: white; font-weight: bold; height: 40px;")
        self.btn_burst.clicked.connect(lambda: self.burst_clicked.emit())
        layout.addWidget(self.btn_burst)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.btn_save_data = QPushButton("Save Dataset (Folder)")
        self.btn_save_data.clicked.connect(lambda: self.save_data_clicked.emit())
        layout.addWidget(self.btn_save_data)

        self.btn_save_mat = QPushButton("Save for MATLAB (.mat)")
        self.btn_save_mat.clicked.connect(lambda: self.save_mat_clicked.emit())
        layout.addWidget(self.btn_save_mat)
        
        layout.addStretch()

    def update_stats(self, vis, sigma, sat_status):
        self.lbl_vis.setText(f"{vis:.3f}")
        self.lbl_sigma.setText(f"{sigma:.1f} um")
        if sat_status:
            self.lbl_sat.setText("SATURATED!")
            self.lbl_sat.setStyleSheet("color: red; font-weight: bold; background-color: yellow;")
        else:
            self.lbl_sat.setText("OK")
            self.lbl_sat.setStyleSheet("color: green; font-weight: bold;")