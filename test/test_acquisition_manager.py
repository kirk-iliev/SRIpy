import sys
import types
import importlib
from dataclasses import dataclass

import numpy as np
import pytest


class DummySignal:
    def __init__(self):
        self._callbacks = []
        self.emitted = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def emit(self, *args, **kwargs):
        self.emitted.append((args, kwargs))
        for cb in self._callbacks:
            cb(*args, **kwargs)


class SignalDescriptor:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        signal = obj.__dict__.get(self.name)
        if signal is None:
            signal = DummySignal()
            obj.__dict__[self.name] = signal
        return signal


class DummyQThread:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True

    def quit(self):
        self.started = False

    def wait(self, _timeout):
        return True


@dataclass
class DummyFitResult:
    success: bool


class DummyFitter:
    def __init__(self, wavelength: float = 550e-9, slit_separation: float = 0.05, distance: float = 16.5, min_signal: float = 50.0):
        self.wavelength = wavelength
        self.slit_sep = slit_separation
        self.distance = distance
        self.min_signal = min_signal


class DummyConfigManager:
    def load(self):
        return {
            "camera": {
                "exposure_ms": 10.0,
                "gain_db": 0.0,
                "transpose": True,
                "saturation_threshold": 100,
            },
            "analysis": {
                "min_signal_threshold": 75.0,
                "autocenter_min_signal": 200.0,
                "analysis_timeout_s": 3.0,
            },
            "burst": {
                "default_frames": 25,
            },
            "physics": {
                "wavelength_nm": 800.0,
                "slit_separation_mm": 1.2,
                "distance_m": 2.5,
            },
            "roi": {
                "rows_min": 10,
                "rows_max": 20,
                "fit_width_min": 30,
                "fit_width_max": 50,
                "auto_center": True,
            },
        }


class DummyDriver:
    def connect(self):
        return None

    def close(self):
        return None


class DummyCameraCommand:
    START_LIVE = object()
    STOP_LIVE = object()
    SET_EXPOSURE = object()
    SET_GAIN = object()
    CAPTURE_BACKGROUND = object()
    START_BURST = object()
    SHUTDOWN = object()


class DummyCameraIoThread:
    def __init__(self, *args, **kwargs):
        return None

    def enqueue(self, *args, **kwargs):
        return None

    def start(self):
        return None

    def wait(self, _timeout):
        return True


def _install_test_stubs(monkeypatch):
    qtcore = types.ModuleType("PyQt6.QtCore")
    qtcore.QObject = object
    qtcore.QThread = DummyQThread
    qtcore.pyqtSignal = lambda *args, **kwargs: SignalDescriptor()

    qt = types.ModuleType("PyQt6")

    fitter_mod = types.ModuleType("analysis.fitter")
    fitter_mod.InterferenceFitter = DummyFitter
    fitter_mod.FitResult = DummyFitResult

    config_mod = types.ModuleType("core.config_manager")
    config_mod.ConfigManager = DummyConfigManager

    manta_mod = types.ModuleType("hardware.manta_driver")
    manta_mod.MantaDriver = DummyDriver

    camera_io_mod = types.ModuleType("hardware.camera_io_thread")
    camera_io_mod.CameraIoThread = DummyCameraIoThread
    camera_io_mod.CameraCommand = DummyCameraCommand

    monkeypatch.setitem(sys.modules, "PyQt6", qt)
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", qtcore)
    monkeypatch.setitem(sys.modules, "analysis.fitter", fitter_mod)
    monkeypatch.setitem(sys.modules, "core.config_manager", config_mod)
    monkeypatch.setitem(sys.modules, "hardware.manta_driver", manta_mod)
    monkeypatch.setitem(sys.modules, "hardware.camera_io_thread", camera_io_mod)
    monkeypatch.setitem(sys.modules, "vmbpy", types.ModuleType("vmbpy"))


@pytest.fixture
def acq_module(monkeypatch):
    _install_test_stubs(monkeypatch)
    module = importlib.import_module("core.acquisition_manager")
    module = importlib.reload(module)
    yield module
    sys.modules.pop("core.acquisition_manager", None)


def test_apply_config_sets_state(acq_module):
    manager = acq_module.AcquisitionManager()

    assert manager.roi_slice == slice(10, 20)
    assert manager.roi_x_limits == (30, 50)
    assert manager.autocenter_enabled is True
    assert manager.transpose_enabled is True
    assert manager.saturation_threshold == 100

    assert manager._autocenter_min_signal == 200.0
    assert manager._analysis_timeout_s == 3.0
    assert manager._default_burst_frames == 25

    assert manager.fitter.min_signal == 75.0
    assert manager.fitter_burst.min_signal == 75.0

    assert manager.fitter.wavelength == pytest.approx(800.0e-9)
    assert manager.fitter.slit_sep == pytest.approx(1.2e-3)
    assert manager.fitter.distance == pytest.approx(2.5)


def test_process_live_frame_autocenter_updates_roi(acq_module):
    manager = acq_module.AcquisitionManager()

    manager.autocenter_enabled = True
    manager._autocenter_min_signal = 200.0
    manager.transpose_enabled = False # Default behavior: integrate rows -> horizontal lineout
    manager.roi_slice = slice(0, 4)
    manager.roi_x_limits = (1, 5)

    # h=4, w=10. Horizontal integration sums along rows.
    # We want a peak at column 7.
    img = np.zeros((4, 10), dtype=np.float32)
    img[:, 7] = 250.0

    manager._process_live_frame(img)

    assert manager.live_data_ready.emitted
    assert manager.roi_updated.emitted
    # peak at 7, width 4 -> [7-2, 7+2] = [5, 9]
    assert manager.roi_x_limits == (5, 9)


def test_process_live_frame_saturation_emits_update(acq_module):
    manager = acq_module.AcquisitionManager()
    manager.saturation_threshold = 100
    manager.roi_slice = slice(0, 2)

    img = np.zeros((4, 10), dtype=np.float32)
    img[0, 0] = 150.0

    manager._process_live_frame(img)

    assert manager.saturation_updated.emitted
    assert manager.last_saturated is True


def test_toggle_background_without_frame_emits_error(acq_module):
    manager = acq_module.AcquisitionManager()

    manager.toggle_background(True)

    assert manager.error_occurred.emitted
    msg = manager.error_occurred.emitted[0][0][0]
    assert msg == "No background frame available."