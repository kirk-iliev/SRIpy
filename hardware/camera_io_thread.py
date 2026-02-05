import logging
import queue
from typing import Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from core.acquisition import BurstWorker


class CameraIoThread(QThread):
    frame_ready = pyqtSignal(object)
    background_ready = pyqtSignal(object)
    burst_progress = pyqtSignal(int)
    burst_finished = pyqtSignal(object)
    burst_error = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, driver, fitter_burst):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._driver = driver
        self._fitter_burst = fitter_burst
        self._cmd_queue: "queue.Queue[tuple[str, tuple]]" = queue.Queue()
        self._live_running = False
        self._shutdown = False

    def enqueue(self, command: str, *args) -> None:
        self._cmd_queue.put((command, args))

    def run(self) -> None:
        while not self._shutdown:
            try:
                command, args = self._cmd_queue.get_nowait()
                self._handle_command(command, args)
            except queue.Empty:
                pass

            if self._live_running:
                try:
                    frame = self._driver.acquire_frame(timeout=0.1)
                    if frame is not None:
                        self.frame_ready.emit(frame)
                except Exception as e:
                    self.error.emit(str(e))
                    self._live_running = False
            else:
                self.msleep(10)

        if self._live_running:
            try:
                self._driver.stop_stream()
            except Exception:
                pass

    def _handle_command(self, command: str, args: tuple) -> None:
        if command == "start_live":
            if not self._live_running:
                try:
                    self._driver.start_stream()
                    self._live_running = True
                except Exception as e:
                    self.error.emit(str(e))
        elif command == "stop_live":
            if self._live_running:
                try:
                    self._driver.stop_stream()
                except Exception as e:
                    self.error.emit(str(e))
                self._live_running = False
        elif command == "set_exposure":
            exposure_ms, = args
            try:
                self._driver.exposure = exposure_ms / 1000.0
            except Exception as e:
                self.error.emit(str(e))
        elif command == "set_gain":
            gain_db, = args
            try:
                self._driver.gain = gain_db
            except Exception as e:
                self.error.emit(str(e))
        elif command == "capture_background":
            was_live = self._live_running
            if was_live:
                try:
                    self._driver.stop_stream()
                except Exception:
                    pass
                self._live_running = False
            try:
                frame = self._driver.acquire_frame(timeout=2.0)
                self.background_ready.emit(frame)
            except Exception as e:
                self.error.emit(str(e))
                self.background_ready.emit(None)
            if was_live:
                try:
                    self._driver.start_stream()
                    self._live_running = True
                except Exception as e:
                    self.error.emit(str(e))
        elif command == "start_burst":
            n_frames, roi_slice, roi_x_limits, transpose_enabled, background = args
            was_live = self._live_running
            if was_live:
                try:
                    self._driver.stop_stream()
                except Exception:
                    pass
                self._live_running = False
            try:
                worker = BurstWorker(
                    self._driver,
                    self._fitter_burst,
                    n_frames,
                    roi_slice,
                    roi_x_limits,
                    transpose=transpose_enabled,
                    background=background,
                )
                worker.progress.connect(self.burst_progress.emit)
                worker.finished.connect(self.burst_finished.emit)
                worker.error.connect(self.burst_error.emit)
                worker.run_burst()
            except Exception as e:
                self.burst_error.emit(str(e))
            if was_live:
                try:
                    self._driver.start_stream()
                    self._live_running = True
                except Exception as e:
                    self.error.emit(str(e))
        elif command == "shutdown":
            self._shutdown = True
