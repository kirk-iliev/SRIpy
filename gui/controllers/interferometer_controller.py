import copy
import numpy as np
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMessageBox, QFileDialog

from core.acquisition_manager import AcquisitionManager
from core.data_model import DataManager, ExperimentMetadata, ExperimentResult

class InterferometerController(QObject):
    def __init__(self, view):
        super().__init__()
        self.view = view
        self.model = AcquisitionManager()
        self._suppress_roi_sync = False
        self._user_is_dragging = False  # Track user interaction state
        
        self._connect_signals()
        
        # Initial Hardware Setup
        try:
            self.model.initialize()
        except RuntimeError as e:
            QMessageBox.critical(self.view, "Connection Error", str(e))

        # Apply config to UI, then push to model
        self._apply_config_to_view()
        self._sync_all_params()
        
        # Update burst button label with configured frame count
        self.view.controls.set_burst_frame_count(self.model._default_burst_frames)

    def _connect_signals(self):
        # --- UI -> Controller -> Model ---
        # Buttons
        self.view.controls.toggle_live_clicked.connect(self._handle_live_toggle)
        self.view.controls.acquire_bg_clicked.connect(self._handle_bg_acquire)
        self.view.controls.burst_clicked.connect(self._handle_burst)
        
        # Value Changes
        self.view.controls.exposure_changed.connect(self.model.set_exposure)
        self.view.controls.gain_changed.connect(self.model.set_gain)
        self.view.controls.chk_transpose.toggled.connect(self._handle_transpose_toggled)
        self.view.controls.chk_bg.toggled.connect(self.model.toggle_background)
        self.view.controls.chk_autocenter.toggled.connect(self.model.set_autocenter)
        
        # Physics Params
        self.view.controls.physics_changed.connect(self._sync_physics)
        
        # ROI Interactions
        self.view.live_widget.roi_changed.connect(self._sync_roi)
        self.view.live_widget.roi_drag_start.connect(self._on_roi_drag_start)
        self.view.live_widget.roi_drag_end.connect(self._on_roi_drag_end)
        
        self.view.controls.reset_roi_clicked.connect(self._reset_roi)

        # Saving
        self.view.controls.save_data_clicked.connect(self._save_dataset)
        self.view.controls.save_mat_clicked.connect(self._save_matlab)
        
        # App Lifecycle
        self.view.close_requested.connect(self.cleanup)

        # --- Model -> Controller -> View ---
        # Data Streams
        self.model.live_data_ready.connect(self._update_display)
        self.model.fit_result_ready.connect(self._update_stats)
        self.model.error_occurred.connect(self._show_error)
        self.model.roi_updated.connect(self._apply_model_roi)
        self.model.saturation_updated.connect(self._update_saturation)
        self.model.background_ready.connect(self._handle_background_ready)
        self.model.live_state_changed.connect(self._handle_live_state)
        
        # Burst Status
        self.model.burst_progress.connect(self.view.controls.progress_bar.setValue)
        self.model.burst_finished.connect(self._handle_burst_finished)
        self.model.burst_error.connect(self._handle_burst_error)
        
        # Load file
        self.view.controls.load_file_clicked.connect(self._handle_load_file)
        self.model.physics_loaded.connect(self._on_physics_loaded)

    # --- Sync Helpers ---

    def _on_physics_loaded(self, wave_nm, slit_mm, dist_m):
        """Updates UI spinners without triggering feedback loops."""
        controls = self.view.controls
        
        # Block signals so updating the UI doesn't accidentally trigger 
        # a "value changed" event that sends data back to the model.
        controls.spin_lambda.blockSignals(True)
        controls.spin_slit.blockSignals(True)
        controls.spin_dist.blockSignals(True)
        
        # Update the values
        controls.spin_lambda.setValue(float(wave_nm))
        controls.spin_slit.setValue(float(slit_mm))
        controls.spin_dist.setValue(float(dist_m))
        
        # Unblock signals
        controls.spin_lambda.blockSignals(False)
        controls.spin_slit.blockSignals(False)
        controls.spin_dist.blockSignals(False)
        
        # Flash the values or log to status bar to show user it happened
        self.view.controls.lbl_sat.setText("PARAMS LOADED")
        self.view.controls.lbl_sat.setStyleSheet("color: blue; font-weight: bold;")


    def _sync_all_params(self):
        self.model.set_exposure(self.view.controls.spin_exp.value())
        self.model.set_gain(self.view.controls.spin_gain.value())
        self.model.set_transpose(self.view.controls.chk_transpose.isChecked())
        self.model.set_autocenter(self.view.controls.chk_autocenter.isChecked())
        self.model.toggle_background(self.view.controls.chk_bg.isChecked())
        self._sync_physics()
        self._sync_roi()

    def _on_roi_drag_start(self):
        self._user_is_dragging = True

    def _on_roi_drag_end(self):
        self._user_is_dragging = False
        self._sync_roi()

    def _sync_roi(self):
        if self._suppress_roi_sync:
            return
        rows = self.view.live_widget.get_roi_rows()
        width = self.view.live_widget.get_roi_width()
        if self.view.controls.chk_transpose.isChecked():
            # Display rows map to original columns; display width maps to original rows
            self.model.set_roi(width[0], width[1], rows[0], rows[1])
        else:
            self.model.set_roi(rows[0], rows[1], width[0], width[1])

    def _reset_roi(self):
        """Reset ROI to default values and DISABLE autocenter."""
        c = self.model.cfg
        self._suppress_roi_sync = True
        try:
            self.view.live_widget.set_roi_rows(c['roi']['rows_min'], c['roi']['rows_max'])
            self.view.live_widget.set_roi_width(c['roi']['fit_width_min'], c['roi']['fit_width_max'])
            self.view.controls.chk_autocenter.setChecked(False)
        finally:
            self._suppress_roi_sync = False
        
        self.model.set_autocenter(False)
        self._sync_roi()

    def _apply_config_to_view(self):
        c = self.model.cfg
        self.view.controls.spin_exp.blockSignals(True)
        self.view.controls.spin_gain.blockSignals(True)
        self.view.controls.chk_transpose.blockSignals(True)
        self.view.controls.chk_autocenter.blockSignals(True)
        self.view.controls.chk_bg.blockSignals(True)

        self.view.controls.spin_exp.setValue(c['camera']['exposure_ms'])
        self.view.controls.spin_gain.setValue(c['camera']['gain_db'])
        self.view.controls.chk_transpose.setChecked(c['camera']['transpose'])

        self.view.controls.spin_lambda.setValue(c['physics']['wavelength_nm'])
        self.view.controls.spin_slit.setValue(c['physics']['slit_separation_mm'])
        self.view.controls.spin_dist.setValue(c['physics']['distance_m'])

        self.view.controls.chk_autocenter.setChecked(c['roi']['auto_center'])

        self.view.controls.chk_bg.setChecked(False)
        self.view.controls.chk_bg.setEnabled(False)

        self.view.controls.spin_exp.blockSignals(False)
        self.view.controls.spin_gain.blockSignals(False)
        self.view.controls.chk_transpose.blockSignals(False)
        self.view.controls.chk_autocenter.blockSignals(False)
        self.view.controls.chk_bg.blockSignals(False)

        self._suppress_roi_sync = True
        try:
            self.view.live_widget.set_roi_rows(c['roi']['rows_min'], c['roi']['rows_max'])
            self.view.live_widget.set_roi_width(c['roi']['fit_width_min'], c['roi']['fit_width_max'])
        finally:
            self._suppress_roi_sync = False


    def _sync_physics(self):
        l = self.view.controls.spin_lambda.value() * 1e-9
        s = self.view.controls.spin_slit.value() * 1e-3
        d = self.view.controls.spin_dist.value()
        self.model.set_physics_params(l, s, d)

    # --- Logic Handlers ---
    def _handle_live_toggle(self, checked):
        if checked:
            self.model.start_live()
        else:
            self.model.stop_live()

    def _handle_live_state(self, is_live: bool) -> None:
        self.view.controls.btn_live.blockSignals(True)
        try:
            self.view.controls.btn_live.setChecked(is_live)
        finally:
            self.view.controls.btn_live.blockSignals(False)

        if is_live:
            self.view.controls.btn_live.setText("Stop Live")
            self.view.controls.btn_live.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        else:
            self.view.controls.btn_live.setText("Start Live")
            self.view.controls.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold;")

    def _handle_bg_acquire(self):
        self.model.capture_background()

    def _handle_background_ready(self, frame):
        if frame is None:
            self.view.controls.chk_bg.setEnabled(False)
            self.view.controls.chk_bg.setChecked(False)
            return
        self.view.controls.chk_bg.setEnabled(True)
        self.view.controls.chk_bg.setChecked(True)

    def _handle_burst(self):
        self.view.controls.progress_bar.setValue(0)
        self.view.controls.progress_bar.setVisible(True)
        
        # Disable ALL acquisition triggers
        self.view.controls.btn_burst.setEnabled(False)
        self.view.controls.btn_live.setEnabled(False)
        self.view.controls.btn_bg.setEnabled(False) # Disable background button too
        
        self.model.start_burst(self.model._default_burst_frames) 

    def _handle_burst_finished(self, res):
        self.view.controls.progress_bar.setVisible(False)
        self.view.controls.btn_burst.setEnabled(True)
        self.view.controls.btn_live.setEnabled(True)
        self.view.controls.btn_bg.setEnabled(True) # Re-enable background button
        
        if self.model.is_live_running():
            self.view.controls.btn_live.setChecked(True)
            self.view.controls.btn_live.setText("Stop Live")
            self.view.controls.btn_live.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        else:
            self.view.controls.btn_live.setChecked(False)
            self.view.controls.btn_live.setText("Start Live")
            self.view.controls.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        
        self.view.history_widget.add_point(res.mean_sigma)
        QMessageBox.information(self.view, "Burst Done", f"Mean Sigma: {res.mean_sigma:.2f} um")

    def _handle_burst_error(self, msg):
        self.view.controls.progress_bar.setVisible(False)
        self.view.controls.btn_burst.setEnabled(True)
        self.view.controls.btn_live.setEnabled(True)
        self.view.controls.btn_bg.setEnabled(True)
        
        if self.model.is_live_running():
            self.view.controls.btn_live.setChecked(True)
            self.view.controls.btn_live.setText("Stop Live")
            self.view.controls.btn_live.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        else:
            self.view.controls.btn_live.setChecked(False)
            self.view.controls.btn_live.setText("Start Live")
            self.view.controls.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        QMessageBox.warning(self.view, "Burst Error", msg)

    def _update_display(self, img, lineout):
        self.view.live_widget.update_image(img)
        self.view.live_widget.update_lineout(np.arange(len(lineout)), lineout)
        self._update_saturation(self.model.last_saturated)
        
    def _update_stats(self, res, x_data):
        # Update Graphs
        if res.success:
            self.view.live_widget.update_fit(x_data, res.fitted_curve)
        else:
            self.view.live_widget.update_fit([], [])

        # Update Text Numbers (Vis, Sigma)
        self.view.controls.lbl_vis.setText(f"{res.visibility:.3f}")
        self.view.controls.lbl_sigma.setText(f"{res.sigma_microns:.1f} um")
        
        # Update Status Label (The Fix)
        is_saturated = self.model.last_saturated
        
        # Saturation Warning (Always show this if bad)
        if is_saturated:
            self.view.controls.lbl_sat.setText("SATURATED!")
            self.view.controls.lbl_sat.setStyleSheet("color: red; font-weight: bold; background-color: yellow;")
        
        # Static File Loaded (If viewing a loaded file)
        elif self.model.is_static_mode():
            self.view.controls.lbl_sat.setText("FILE LOADED")
            self.view.controls.lbl_sat.setStyleSheet("color: blue; font-weight: bold;")
            
        # Normal Live Status
        else:
            self.view.controls.lbl_sat.setText("OK")
            self.view.controls.lbl_sat.setStyleSheet("color: green; font-weight: bold;")

        # Update History Plot
        if res.success and self.view.tabs.currentIndex() == 1:
            self.view.history_widget.add_point(res.sigma_microns)

    def _update_saturation(self, is_saturated: bool):
        if is_saturated:
            self.view.controls.lbl_sat.setText("SATURATED!")
            self.view.controls.lbl_sat.setStyleSheet(
                "color: red; font-weight: bold; background-color: yellow;"
            )
        else:
            self.view.controls.lbl_sat.setText("OK")
            self.view.controls.lbl_sat.setStyleSheet("color: green; font-weight: bold;")

    def _show_error(self, msg):
        """Display error to user via console and status bar."""
        import logging
        logging.getLogger(__name__).error(f"Error: {msg}")
        self.view.controls.lbl_sat.setText("ERROR")
        self.view.controls.lbl_sat.setStyleSheet(
            "color: white; font-weight: bold; background-color: red;"
        ) 

    def _handle_transpose_toggled(self, checked: bool):
        self.model.set_transpose(checked)
        # Refresh ROI display to match orientation
        self._apply_model_roi(
            self.model.roi_slice.start,
            self.model.roi_slice.stop,
            self.model.roi_x_limits[0],
            self.model.roi_x_limits[1],
        )

    def _apply_model_roi(self, y_min, y_max, x_min, x_max):
        if self._user_is_dragging:
            return

        self._suppress_roi_sync = True
        try:
            if self.view.controls.chk_transpose.isChecked():
                # Display rows map to original columns; display width maps to original rows
                self.view.live_widget.set_roi_rows(x_min, x_max)
                self.view.live_widget.set_roi_width(y_min, y_max)
            else:
                self.view.live_widget.set_roi_rows(y_min, y_max)
                self.view.live_widget.set_roi_width(x_min, x_max)
        finally:
            self._suppress_roi_sync = False

    def _save_dataset(self):
        meta = ExperimentMetadata(
            exposure_s=self.view.controls.spin_exp.value() / 1000.0,
            gain_db=self.view.controls.spin_gain.value(),
            wavelength_nm=self.view.controls.spin_lambda.value(),
            slit_separation_mm=self.view.controls.spin_slit.value(),
            distance_m=self.view.controls.spin_dist.value(),
        )
        
        if self.model.last_raw_image is None or self.model.last_fit_result is None:
            QMessageBox.warning(self.view, "Save Error", "No data available to save.")
            return

        res = ExperimentResult(
            visibility=self.model.last_fit_result.visibility,
            sigma_microns=self.model.last_fit_result.sigma_microns,
            lineout_y=self.model.last_lineout.tolist() if isinstance(self.model.last_lineout, np.ndarray) else self.model.last_lineout,
            fit_y=self.model.last_fit_result.fitted_curve.tolist() if hasattr(self.model.last_fit_result, 'fitted_curve') and isinstance(self.model.last_fit_result.fitted_curve, np.ndarray) else self.model.last_fit_result.fitted_curve,
            is_saturated=self.model.last_saturated
        )

        dir_path = QFileDialog.getExistingDirectory(self.view, "Select Save Directory")
        if dir_path:
            path = DataManager.save_dataset(dir_path, "SRI_Data", self.model.last_raw_image, meta, res)
            QMessageBox.information(self.view, "Saved", f"Saved to:\n{path}")

    def _save_matlab(self):
        meta = ExperimentMetadata(
            exposure_s=self.view.controls.spin_exp.value() / 1000.0,
            gain_db=self.view.controls.spin_gain.value(),
            wavelength_nm=self.view.controls.spin_lambda.value(),
            slit_separation_mm=self.view.controls.spin_slit.value(),
            distance_m=self.view.controls.spin_dist.value(),
        )

        if self.model.last_raw_image is None or self.model.last_fit_result is None:
            QMessageBox.warning(self.view, "Save Error", "No data available to save.")
            return

        res = ExperimentResult(
            visibility=self.model.last_fit_result.visibility,
            sigma_microns=self.model.last_fit_result.sigma_microns,
            lineout_y=self.model.last_lineout.tolist() if isinstance(self.model.last_lineout, np.ndarray) else self.model.last_lineout,
            fit_y=self.model.last_fit_result.fitted_curve.tolist() if hasattr(self.model.last_fit_result, 'fitted_curve') and isinstance(self.model.last_fit_result.fitted_curve, np.ndarray) else self.model.last_fit_result.fitted_curve,
            is_saturated=self.model.last_saturated
        )

        dir_path = QFileDialog.getExistingDirectory(self.view, "Select Save Directory")
        if dir_path:
            path = DataManager.save_matlab(dir_path, "SRI_Matlab", self.model.last_raw_image, meta, res)
            QMessageBox.information(self.view, "Saved", f"Saved .mat file:\n{path}")

    def _handle_load_file(self):
        """Pauses live view and loads a static file."""
        # Force Live View to stop so it doesn't overwrite the loaded file
        if self.model.is_live_running():
            self.model.stop_live()
            # Update UI to reflect stopped state
            self.view.controls.btn_live.setChecked(False)
            self.view.controls.btn_live.setText("Start Live")
            self.view.controls.btn_live.setStyleSheet("background-color: green; color: white; font-weight: bold;")

        # Open File Dialog
        # Support generic images AND .mat files
        file_filter = "All Supported (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.mat);;Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Matlab Data (*.mat)"
        path, _ = QFileDialog.getOpenFileName(self.view, "Load Beam Image", "", file_filter)
        
        if not path:
            return

        # Pass to Model
        try:
            self.model.load_static_frame(path)
            self.view.controls.lbl_sat.setText("FILE LOADED")
            self.view.controls.lbl_sat.setStyleSheet("color: blue; font-weight: bold;")
        except Exception as e:
            QMessageBox.warning(self.view, "Load Error", f"Could not load file:\n{str(e)}")

    def _save_current_config(self):
        """Scrape UI settings and save to disk via ConfigManager."""
        rows = self.view.live_widget.get_roi_rows()
        width = self.view.live_widget.get_roi_width()
        if self.view.controls.chk_transpose.isChecked():
            rows_min, rows_max = width[0], width[1]
            fit_width_min, fit_width_max = rows[0], rows[1]
        else:
            rows_min, rows_max = rows[0], rows[1]
            fit_width_min, fit_width_max = width[0], width[1]

        ui_config = copy.deepcopy(self.model.cfg)

        ui_config["camera"] = {
            "exposure_ms": self.view.controls.spin_exp.value(),
            "gain_db": self.view.controls.spin_gain.value(),
            "transpose": self.view.controls.chk_transpose.isChecked(),
            "subtract_background": self.view.controls.chk_bg.isChecked(),
            "saturation_threshold": self.model.saturation_threshold,
        }
        ui_config["physics"] = {
            "wavelength_nm": self.view.controls.spin_lambda.value(),
            "slit_separation_mm": self.view.controls.spin_slit.value(),
            "distance_m": self.view.controls.spin_dist.value(),
        }
        ui_config["roi"] = {
            "rows_min": rows_min,
            "rows_max": rows_max,
            "fit_width_min": fit_width_min,
            "fit_width_max": fit_width_max,
            "auto_center": self.view.controls.chk_autocenter.isChecked(),
        }
        ui_config.setdefault("analysis", {})
        ui_config.setdefault("burst", {})
        ui_config["analysis"]["min_signal_threshold"] = self.model.fitter.min_signal
        ui_config["analysis"]["autocenter_min_signal"] = self.model._autocenter_min_signal
        ui_config["analysis"]["analysis_timeout_s"] = self.model._analysis_timeout_s
        ui_config["burst"]["default_frames"] = self.model._default_burst_frames

        self.model.config_manager.save(ui_config)

    def cleanup(self, event):
        # Save settings before shutting down, but ensure shutdown happens
        try:
            self._save_current_config()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save config on exit: {e}")
        finally:
            self.model.shutdown()
        event.accept()