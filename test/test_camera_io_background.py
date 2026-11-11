"""
Regression test: background capture must work with a driver that, like the
real MantaDriver, refuses to return frames unless streaming is active.
"""

import numpy as np
import pytest

from hardware.camera_io_thread import CameraIoThread, CameraCommand
from conftest import MockDriver


class StrictDriver(MockDriver):
    """Mimics MantaDriver.acquire_frame: raises if the stream isn't running."""

    def acquire_frame(self, timeout=1.0):
        if not self.is_streaming:
            raise RuntimeError("Streaming not started. Call start_stream() first.")
        return super().acquire_frame(timeout)


@pytest.mark.parametrize("was_live", [False, True])
def test_capture_background_with_strict_driver(was_live):
    bg = np.full((4, 4), 7, dtype=np.uint16)
    driver = StrictDriver(frames=[bg])
    thread = CameraIoThread(driver, fitter_burst=None)

    backgrounds, errors, live_states = [], [], []
    thread.background_ready.connect(backgrounds.append)
    thread.error.connect(errors.append)
    thread.live_state_changed.connect(live_states.append)

    if was_live:
        driver.start_stream()
        thread._live_running = True

    thread._handle_command(CameraCommand.CAPTURE_BACKGROUND, ())

    assert errors == []
    assert len(backgrounds) == 1 and backgrounds[0] is not None
    np.testing.assert_array_equal(backgrounds[0], bg)
    # Camera ends in the same streaming state it started in
    assert driver.is_streaming is was_live
    assert thread._live_running is was_live
    if was_live:
        assert live_states == [False, True]
