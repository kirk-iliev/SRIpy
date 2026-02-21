import os
import sys
import queue
import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass

# Ensure project root is on sys.path so imports like `core.acquisition` work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.acquisition import BurstWorker


@dataclass
class FakeFitResult:
    """Mock fit result for testing."""
    success: bool = True
    visibility: float = 0.5
    sigma_microns: float = 1.0
    raw_visibility: float = 0.5
    max_intensity: float = 1000.0
    min_intensity: float = 100.0


class FakeFitter:
    """Mock fitter that returns deterministic results."""
    def fit(self, y):
        # Return a result compatible with BurstWorker expectations
        return FakeFitResult(
            success=True,
            visibility=0.5,
            sigma_microns=1.0,
            raw_visibility=0.5,
            max_intensity=float(np.max(y)) if len(y) > 0 else 1000.0,
            min_intensity=float(np.min(y)) if len(y) > 0 else 100.0,
        )


def test_progress_with_dropped_frames():
    """Test that BurstWorker correctly handles dropped frames and emits progress updates."""
    # Create test frames
    frame1 = np.ones((4, 4), dtype=np.uint16) * 100
    frame2 = np.ones((4, 4), dtype=np.uint16) * 200
    frame3 = np.ones((4, 4), dtype=np.uint16) * 300

    # Populate frame queue: present, dropped (None), present, dropped, present
    frame_queue = queue.Queue()
    for frame in [frame1, None, frame2, None, frame3]:
        frame_queue.put(frame)

    fitter = FakeFitter()

    # Create worker with roi_x_map to slice the lineout
    worker = BurstWorker(
        frame_queue=frame_queue,
        fitter=fitter,
        n_frames=5,
        roi_slice=slice(0, 4),
        roi_x_map=(0, 4),
        transpose=False,
        background=None
    )

    # Track signals
    progress_updates = []
    results = []

    worker.progress.connect(lambda v: progress_updates.append(v))
    worker.finished.connect(lambda r: results.append(r))

    # Run the burst
    worker.run_burst()

    # Verify progress emitted
    assert progress_updates, "No progress updates emitted"
    assert progress_updates[-1] == 100, f"Final progress not 100: {progress_updates}"
    assert all(progress_updates[i] <= progress_updates[i+1] for i in range(len(progress_updates)-1)), \
        "Progress decreased at some point"

    # Verify result
    assert results, "No burst result emitted"
    result = results[0]
    assert result.n_frames == 5, f"Expected n_frames=5, got {result.n_frames}"
    assert len(result.lineout_history) == 3, f"Expected 3 captured frames, got {len(result.lineout_history)}"
    assert len(result.vis_history) == 3, f"Expected 3 fit results, got {len(result.vis_history)}"
    assert any(v >= 50 for v in progress_updates), "No acquisition phase progress emitted"



def test_no_frames_captured_emits_immediate_100():
    """Test that when no frames are captured, progress immediately reaches 100."""
    # Empty queue: all None frames
    frame_queue = queue.Queue()
    for _ in range(2):
        frame_queue.put(None)

    fitter = FakeFitter()
    worker = BurstWorker(
        frame_queue=frame_queue,
        fitter=fitter,
        n_frames=2,
        roi_slice=slice(0, 2),
        roi_x_map=(0, 2),
        transpose=False,
        background=None
    )

    progress_updates = []
    results = []

    worker.progress.connect(lambda v: progress_updates.append(v))
    worker.finished.connect(lambda r: results.append(r))

    worker.run_burst()

    # When no frames captured, should emit 100 for phase 2 (analysis)
    assert progress_updates, "No progress updates emitted"
    assert progress_updates[-1] == 100, f"Final progress not 100: {progress_updates}"

    # Result should be empty
    assert results, "No burst result emitted"
    result = results[0]
    assert len(result.lineout_history) == 0, "Expected 0 lineouts"


def test_burst_accumulates_statistics():
    """Test that BurstWorker correctly accumulates visibility statistics."""
    # Create consistent frames that will yield predictable results
    frame = np.ones((10, 10), dtype=np.uint16) * 500

    frame_queue = queue.Queue()
    for _ in range(3):  # 3 frames
        frame_queue.put(frame.copy())

    fitter = FakeFitter()
    worker = BurstWorker(
        frame_queue=frame_queue,
        fitter=fitter,
        n_frames=3,
        roi_slice=slice(0, 10),
        roi_x_map=(0, 10),
        transpose=False,
        background=None
    )

    results = []
    worker.finished.connect(lambda r: results.append(r))

    worker.run_burst()

    assert results, "No burst result emitted"
    result = results[0]
    assert result.n_frames == 3
    assert len(result.lineout_history) == 3
    assert len(result.vis_history) == 3
    assert result.mean_visibility == 0.5, "Expected mean_visibility=0.5"
    assert result.mean_sigma == 1.0, "Expected mean_sigma=1.0"
