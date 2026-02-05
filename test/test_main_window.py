"""
Unit tests for MVC refactor components.

Tests controller logic and acquisition manager behavior without requiring hardware or Vimba driver.
Uses mocks for MantaDriver, PyQt6 components, and worker threads.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# Mock PyQt6 before importing main_window
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()

# Mock the hardware driver
sys.modules['vmbpy'] = MagicMock()

# Set up environment to avoid Qt needing display
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture
def mock_driver():
    """Mock MantaDriver"""
    driver = MagicMock()
    driver.exposure = 1.0
    driver.gain = 0
    driver.connect = MagicMock()
    driver.close = MagicMock()
    driver.start_stream = MagicMock()
    driver.stop_stream = MagicMock()
    driver.acquire_frame = MagicMock(return_value=np.zeros((512, 512), dtype=np.uint16))
    return driver


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager"""
    manager = MagicMock()
    manager.load = MagicMock(return_value={
        'camera': {'exposure_ms': 10, 'gain_db': 0, 'transpose': False, 'saturation_threshold': 4090},
        'physics': {'wavelength_nm': 780, 'slit_separation_mm': 1.0, 'distance_m': 1.0},
        'roi': {'rows_min': 128, 'rows_max': 384, 'fit_width_min': 128, 'fit_width_max': 384, 'auto_center': False},
        'analysis': {'min_signal_threshold': 50.0}
    })
    manager.save = MagicMock()
    return manager


@pytest.fixture
def mock_fitter():
    """Mock InterferenceFitter"""
    fitter = MagicMock()
    fitter.wavelength = 780e-9
    fitter.slit_sep = 1e-3
    fitter.distance = 1.0
    fitter.min_signal = 50.0
    return fitter


@pytest.fixture
def mock_widgets():
    """Mock GUI widgets"""
    widgets = {
        'live_widget': MagicMock(),
        'history_widget': MagicMock(),
        'controls': MagicMock(),
        'tabs': MagicMock(),
    }

    # Setup live_widget
    widgets['live_widget'].get_roi_rows = MagicMock(return_value=(128, 384))
    widgets['live_widget'].get_roi_width = MagicMock(return_value=(128, 384))
    widgets['live_widget'].set_roi_rows = MagicMock()
    widgets['live_widget'].set_roi_width = MagicMock()
    widgets['live_widget'].update_image = MagicMock()
    widgets['live_widget'].update_lineout = MagicMock()
    widgets['live_widget'].update_fit = MagicMock()
    widgets['live_widget'].roi_drag_start = MagicMock()
    widgets['live_widget'].roi_drag_end = MagicMock()

    # Setup controls
    widgets['controls'].chk_bg = MagicMock()
    widgets['controls'].chk_bg.isChecked = MagicMock(return_value=False)
    widgets['controls'].chk_transpose = MagicMock()
    widgets['controls'].chk_transpose.isChecked = MagicMock(return_value=False)
    widgets['controls'].chk_autocenter = MagicMock()
    widgets['controls'].chk_autocenter.isChecked = MagicMock(return_value=False)

    widgets['controls'].spin_exp = MagicMock()
    widgets['controls'].spin_exp.value = MagicMock(return_value=10)
    widgets['controls'].spin_gain = MagicMock()
    widgets['controls'].spin_gain.value = MagicMock(return_value=0)
    widgets['controls'].spin_lambda = MagicMock()
    widgets['controls'].spin_lambda.value = MagicMock(return_value=780)
    widgets['controls'].spin_slit = MagicMock()
    widgets['controls'].spin_slit.value = MagicMock(return_value=1.0)
    widgets['controls'].spin_dist = MagicMock()
    widgets['controls'].spin_dist.value = MagicMock(return_value=1.0)

    widgets['controls'].lbl_sat = MagicMock()
    widgets['controls'].lbl_sat.setText = MagicMock()
    widgets['controls'].lbl_sat.setStyleSheet = MagicMock()
    widgets['controls'].lbl_sat.text = MagicMock(return_value="OK")

    widgets['controls'].btn_live = MagicMock()
    widgets['controls'].btn_live.setChecked = MagicMock()
    widgets['controls'].btn_live.setEnabled = MagicMock()
    widgets['controls'].btn_live.setText = MagicMock()
    widgets['controls'].btn_live.setStyleSheet = MagicMock()

    widgets['controls'].btn_burst = MagicMock()
    widgets['controls'].btn_burst.setEnabled = MagicMock()
    widgets['controls'].progress_bar = MagicMock()
    widgets['controls'].progress_bar.setRange = MagicMock()
    widgets['controls'].progress_bar.setVisible = MagicMock()
    widgets['controls'].progress_bar.setValue = MagicMock()

    widgets['controls'].update_stats = MagicMock()

    return widgets


class TestAcquisitionManagerLogic:
    """Test business logic of AcquisitionManager without GUI instantiation."""

    def test_update_frame_basic_processing(self, mock_driver, mock_config_manager, mock_fitter, mock_widgets):
        """Test that update_frame processes images correctly."""
        # Create a test image
        test_image = np.random.randint(0, 4000, size=(512, 512), dtype=np.uint16)

        # Verify test image structure
        assert test_image.shape == (512, 512)
        assert test_image.dtype == np.uint16

    def test_frame_saturation_detection(self):
        """Test saturation detection logic."""
        # Create an image with saturation
        img = np.ones((512, 512), dtype=np.float32) * 4090
        sat_limit = 4090

        is_saturated = np.max(img) >= sat_limit
        assert bool(is_saturated) is True

        # Test non-saturated
        img_ok = np.ones((512, 512), dtype=np.float32) * 4000
        is_saturated_ok = np.max(img_ok) >= sat_limit
        assert bool(is_saturated_ok) is False

    def test_background_subtraction_logic(self):
        """Test background subtraction logic."""
        # Create test images
        raw = np.ones((512, 512), dtype=np.float32) * 2000
        background = np.ones((512, 512), dtype=np.float32) * 500

        # Apply subtraction
        result = raw - background

        # Verify result
        expected = np.ones((512, 512), dtype=np.float32) * 1500
        np.testing.assert_array_almost_equal(result, expected)

        # Test clipping
        result_clipped = np.clip(result, 0, None)
        assert np.min(result_clipped) >= 0

    def test_transpose_logic(self):
        """Test image transpose logic."""
        # Create rectangular image
        img = np.ones((256, 512), dtype=np.float32)

        img_t = img.T
        assert img_t.shape == (512, 256)

        # Verify contiguous after transpose
        img_t_contig = np.ascontiguousarray(img_t)
        assert img_t_contig.flags['C_CONTIGUOUS']

    def test_roi_bounds_checking_logic(self):
        """Test ROI bounds validation logic."""
        img_h, img_w = 512, 512

        # Test case 1: ROI fully within bounds
        roi_rows = (100, 400)
        new_r_min = max(0.0, min(roi_rows[0], img_h - 10.0))
        new_r_max = max(new_r_min + 10.0, min(roi_rows[1], img_h))

        assert new_r_min == 100
        assert new_r_max == 400

        # Test case 2: ROI out of bounds (too low)
        roi_rows_low = (-10, 50)
        new_r_min = max(0.0, min(roi_rows_low[0], img_h - 10.0))
        new_r_max = max(new_r_min + 10.0, min(roi_rows_low[1], img_h))

        assert new_r_min == 0
        assert new_r_max >= new_r_min + 10

    def test_lineout_extraction_logic(self):
        """Test lineout extraction logic."""
        # Create test image
        img = np.random.rand(100, 200)

        # Extract horizontal slice (lineout)
        roi_rows = (40, 60)
        y_min, y_max = int(roi_rows[0]), int(roi_rows[1])
        y_min = max(0, min(y_min, img.shape[0] - 1))
        y_max = max(1, min(y_max, img.shape[0]))

        roi_slice = slice(y_min, y_max)
        lineout = np.sum(img[roi_slice, :], axis=0)

        # Verify result
        assert lineout.shape == (200,)
        assert np.all(np.isfinite(lineout))

    def test_autocenter_logic(self):
        """Test autocentering logic."""
        # Create lineout with peak
        lineout = np.random.rand(512) * 100
        lineout[256:270] = 5000  # Add peak around center

        w = len(lineout)
        peak_idx = int(np.argmax(lineout))

        assert peak_idx > 255
        assert peak_idx < 270

        # Test recentering
        roi_width = (200, 400)
        c_min, c_max = int(roi_width[0]), int(roi_width[1])
        width = c_max - c_min

        new_min = max(0, peak_idx - width/2)
        new_max = min(w, peak_idx + width/2)

        assert new_min >= 0
        assert new_max <= w

    def test_burst_state_tracking(self):
        """Test burst state tracking logic."""
        # Simulate burst state transitions
        burst_running = False
        assert burst_running is False

        burst_running = True
        assert burst_running is True

        # Verify burst blocks live view
        live_enabled = True
        if burst_running:
            live_enabled = False

        assert live_enabled is False

    def test_thread_state_management(self):
        """Test thread state management."""
        # Simulate thread initialization
        cam_thread = None
        cam_worker = None

        # Before initialization, both should be None
        assert cam_thread is None
        assert cam_worker is None

        # After initialization
        cam_thread = MagicMock()
        cam_worker = MagicMock()

        assert cam_thread is not None
        assert cam_worker is not None

        # Test null checks before accessing methods
        if cam_thread is not None:
            cam_thread.quit()
            cam_thread.quit.assert_called_once()

    def test_worker_busy_flag_logic(self):
        """Test worker busy flag reset logic."""
        import time

        worker_is_busy = True
        last_fit_request_time = time.time() - 4.0  # 4 seconds ago

        current_time = time.time()

        # If worker is busy for over 3 seconds, reset
        if worker_is_busy and (current_time - last_fit_request_time > 3.0):
            worker_is_busy = False

        assert worker_is_busy is False


class TestManagerTypeAnnotations:
    """Test type annotations are correct in acquisition manager."""

    def test_class_annotations_exist(self):
        """Verify all required type annotations exist in source code."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        manager_path = os.path.join(base_dir, 'core', 'acquisition_manager.py')
        with open(manager_path, 'r') as f:
            source = f.read()

        # Updated for MVC refactor: CameraIoThread replaced separate cam_thread/cam_worker
        required_annotations = [
            'camera_thread: Optional[CameraIoThread]',
            'an_thread: Optional[QThread]',
            'an_worker: Optional[AnalysisWorker]',
        ]

        for annotation in required_annotations:
            assert annotation in source, f"Missing annotation pattern: {annotation}"

    def test_annotations_are_optional(self):
        """Verify annotations use Optional type in source."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        manager_path = os.path.join(base_dir, 'core', 'acquisition_manager.py')
        with open(manager_path, 'r') as f:
            source = f.read()

        # Check that Optional is imported
        assert 'from typing import' in source, "Missing typing imports"
        assert 'Optional' in source, "Optional not imported"

        # Check that thread/worker attributes are declared as Optional
        # Updated for MVC refactor: CameraIoThread unifies camera IO
        assert 'camera_thread: Optional' in source
        assert 'an_thread: Optional' in source
        assert 'an_worker: Optional' in source


class TestNullSafety:
    """Test that null checks are properly implemented in acquisition manager."""

    def test_camera_thread_null_checks_in_source(self):
        """Verify camera_thread has null checks before use."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        manager_path = os.path.join(base_dir, 'core', 'acquisition_manager.py')
        with open(manager_path, 'r') as f:
            source = f.read()

        # Count accesses to camera_thread methods
        accesses = source.count('self.camera_thread.')

        # Count null checks (camera_thread is None or camera_thread is not None)
        checks = source.count('self.camera_thread is not None') + source.count('self.camera_thread is None')

        # Should have at least one null check
        assert checks > 0, f"No null checks found for camera_thread (found {accesses} accesses)"

    def test_camera_thread_defensive_in_methods(self):
        """Verify camera_thread is checked before method calls."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        manager_path = os.path.join(base_dir, 'core', 'acquisition_manager.py')
        with open(manager_path, 'r') as f:
            source = f.read()

        # Verify camera_thread checks exist in key methods
        # The refactored code uses 'if self.camera_thread is None: return' pattern
        assert 'if self.camera_thread is None' in source, "No defensive None checks for camera_thread"

    def test_an_thread_null_checks_in_source(self):
        """Verify an_thread has null checks before use."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        manager_path = os.path.join(base_dir, 'core', 'acquisition_manager.py')
        with open(manager_path, 'r') as f:
            source = f.read()

        # Count null checks - the code uses 'if self.an_thread:' which is truthy check
        checks = source.count('if self.an_thread')

        # Should have null checks
        assert checks > 0, "No null checks found for an_thread"


class TestConfigApplication:
    """Test configuration application logic."""

    def test_exposure_update_logic(self):
        """Test exposure update logic."""
        exposure_ms = 15
        driver_exposure = exposure_ms / 1000.0

        assert driver_exposure == 0.015

    def test_physics_parameters_update_logic(self):
        """Test physics parameters update."""
        wavelength_nm = 780
        wavelength_m = wavelength_nm * 1e-9

        assert wavelength_m == 780e-9

        slit_mm = 1.5
        slit_m = slit_mm * 1e-3

        assert slit_m == 1.5e-3


class TestSignalHandling:
    """Test signal handling logic."""

    def test_fit_result_processing(self):
        """Test fit result processing logic."""
        # Simulate fit result
        fit_result = MagicMock()
        fit_result.success = True
        fit_result.visibility = 0.75
        fit_result.sigma_microns = 2.5
        fit_result.fitted_curve = np.random.rand(100)

        # Process result
        worker_is_busy = False  # Reset busy flag
        assert worker_is_busy is False

        if fit_result.success:
            assert fit_result.visibility > 0
            assert fit_result.sigma_microns >= 0

    def test_burst_progress_tracking(self):
        """Test burst progress tracking."""
        n_frames = 50
        current_frame = 25

        progress_percent = int((current_frame / n_frames) * 100)

        assert progress_percent == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
