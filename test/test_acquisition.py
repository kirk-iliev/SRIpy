import os
import sys
import numpy as np
from types import SimpleNamespace
# Ensure project root is on sys.path so imports like `core.acquisition` work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.acquisition import BurstWorker


def make_fake_driver(frames):
    class FakeDriver:
        def __init__(self, frames):
            # make copies so tests can't accidentally mutate shared arrays
            self.frames = [np.array(f) if f is not None else None for f in frames]
        def acquire_frame(self, timeout=1.0):
            if not self.frames:
                return None
            return self.frames.pop(0)
        def start_stream(self):
            pass
        def stop_stream(self):
            pass
    return FakeDriver(frames)


class FakeFitter:
    def get_lineout(self, image, roi_slice=None):
        # Mimic the real behaviour (sum over rows for 2D, passthrough for 1D)
        if image is None:
            return np.array([])
        arr = np.array(image)
        if arr.ndim > 1:
            return np.sum(arr, axis=0)
        return arr

    def fit(self, y):
        # Return a lightweight object compatible with BurstWorker expectations
        return SimpleNamespace(success=True, visibility=0.5, sigma_microns=1.0)


def test_progress_with_dropped_frames():
    # Frames: present, dropped, present, dropped, present
    frames = [np.ones((4, 4)) * i for i in (1, 2, 3)]
    sequence = [frames[0], None, frames[1], None, frames[2]]

    driver = make_fake_driver(sequence)
    fitter = FakeFitter()

    worker = BurstWorker(driver, fitter, n_frames=len(sequence), roi_slice=None, roi_x_map=None)

    progress_updates = []
    result_container = {}

    worker.progress.connect(lambda v: progress_updates.append(v))
    worker.finished.connect(lambda r: result_container.update({'res': r}))

    worker.run_burst()

    # There should be some progress updates and we must end at 100
    assert progress_updates, "No progress updates emitted"
    assert progress_updates[-1] == 100, f"Final progress not 100: {progress_updates}"

    # Progress should be non-decreasing
    assert all(earlier <= later for earlier, later in zip(progress_updates, progress_updates[1:])), "Progress decreased at some point"

    # Verify results: we captured exactly 3 frames (non-None entries)
    result = result_container.get('res')
    assert result is not None
    assert len(result.lineout_history) == 3
    assert result.n_frames == len(sequence)

    # Ensure we emitted progress values that cross the 50% threshold (processing phase)
    assert any(v >= 50 for v in progress_updates), "No processing progress emitted (<50)"


def test_no_frames_captured_emits_immediate_100():
    sequence = [None, None]
    driver = make_fake_driver(sequence)
    fitter = FakeFitter()

    worker = BurstWorker(driver, fitter, n_frames=len(sequence), roi_slice=None, roi_x_map=None)

    progress_updates = []
    worker.progress.connect(lambda v: progress_updates.append(v))

    worker.run_burst()

    # When no frames captured, progress should immediately report completion (100)
    assert progress_updates == [100] or (progress_updates and progress_updates[-1] == 100)
