import logging
import queue
from enum import Enum, auto
from typing import Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from core.acquisition import BurstWorker


class CameraCommand(Enum):
    START_LIVE = auto()
    STOP_LIVE = auto()
    SET_EXPOSURE = auto()
    SET_GAIN = auto()
    CAPTURE_BACKGROUND = auto()
    START_BURST = auto()
    BURST_COMPLETED = auto() # Internal: Signal that burst thread finished
    SHUTDOWN = auto()


class CameraIoThread(QThread):
    frame_ready = pyqtSignal(object)
    background_ready = pyqtSignal(object)
    live_state_changed = pyqtSignal(bool)
    burst_progress = pyqtSignal(int)
    burst_finished = pyqtSignal(object)
    burst_error = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, driver, fitter_burst):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._driver = driver
        self._fitter_burst = fitter_burst
        self._cmd_queue: "queue.Queue[tuple[CameraCommand, tuple]]" = queue.Queue()
        self._live_running = False
        self._shutdown = False
        self._burst_running = False
        self._resume_live_after_burst = False
        self._burst_thread: Optional[QThread] = None
        self._burst_worker: Optional[BurstWorker] = None
        self._burst_id = 0  # Track burst generations to prevent signal crossover

    def enqueue(self, command: CameraCommand, *args) -> None:
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
                    else:
                        self.msleep(1)
                except Exception as e:
                    self.error.emit(str(e))
                    self._live_running = False
                    self.live_state_changed.emit(False)
                    self.msleep(1)
            else:
                self.msleep(10)

        # Ensure burst thread completes before exiting
        self._wait_for_burst_thread()

        if self._live_running:
            try:
                self._driver.stop_stream()
            except Exception:
                pass

    def _handle_command(self, command: CameraCommand, args: tuple) -> None:
        # Guard clause: Ignore commands that conflict with a running burst.
        # EXCEPTION: We allow START_BURST to pass through so we can handle "restart" logic gracefully.
        if self._burst_running and command not in (CameraCommand.SHUTDOWN, CameraCommand.BURST_COMPLETED, CameraCommand.START_BURST):
            if command == CameraCommand.START_LIVE:
                self._resume_live_after_burst = True
                return
            if command == CameraCommand.STOP_LIVE:
                self._resume_live_after_burst = False
                return
            if command in (CameraCommand.SET_EXPOSURE, CameraCommand.SET_GAIN):
                # Allow parameter changes during burst
                pass
            elif command in (CameraCommand.CAPTURE_BACKGROUND,):
                self.error.emit("Command ignored: burst in progress")
                return

        if command == CameraCommand.BURST_COMPLETED:
            # Only clean up if this signal matches the currently running burst ID
            # This prevents delayed signals from old bursts killing a NEW burst
            bid, = args
            if bid == self._burst_id:
                self._cleanup_burst()
            return

        if command == CameraCommand.START_LIVE:
            if not self._live_running:
                try:
                    self._driver.start_stream()
                    self._live_running = True
                    self.live_state_changed.emit(True)
                except Exception as e:
                    self.error.emit(str(e))
        elif command == CameraCommand.STOP_LIVE:
            if self._live_running:
                try:
                    self._driver.stop_stream()
                except Exception as e:
                    self.error.emit(str(e))
                self._live_running = False
                self.live_state_changed.emit(False)
        elif command == CameraCommand.SET_EXPOSURE:
            exposure_ms, = args
            try:
                self._driver.exposure = exposure_ms / 1000.0
            except Exception as e:
                self.error.emit(str(e))
        elif command == CameraCommand.SET_GAIN:
            gain_db, = args
            try:
                self._driver.gain = gain_db
            except Exception as e:
                self.error.emit(str(e))
        elif command == CameraCommand.CAPTURE_BACKGROUND:
            was_live = self._live_running
            if was_live:
                try:
                    self._driver.stop_stream()
                except Exception:
                    pass
                self._live_running = False
                self.live_state_changed.emit(False)
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
                    self.live_state_changed.emit(True)
                except Exception as e:
                    self.error.emit(str(e))
        elif command == CameraCommand.START_BURST:
            # Increment ID immediately. This invalidates any pending BURST_COMPLETED signals 
            # from previous bursts, so they won't delete our new thread.
            self._burst_id += 1
            
            # If a burst is technically "running" (e.g. just finishing cleanup),
            # this will wait for it to die completely before starting the new one.
            self._wait_for_burst_thread()
            self._cleanup_burst() # Ensure clean slate

            n_frames, roi_slice, roi_x_limits, transpose_enabled, background = args
            was_live = self._live_running
            if was_live:
                try:
                    self._driver.stop_stream()
                except Exception:
                    pass
                self._live_running = False
                self.live_state_changed.emit(False)
            try:
                self._start_burst_thread(
                    n_frames,
                    roi_slice,
                    roi_x_limits,
                    transpose_enabled,
                    background,
                    was_live,
                )
            except Exception as e:
                self.burst_error.emit(str(e))
                self._cleanup_burst()
        elif command == CameraCommand.SHUTDOWN:
            self._shutdown = True
            self._resume_live_after_burst = False
            # Wait will occur after breaking the loop

    def _start_burst_thread(
        self,
        n_frames,
        roi_slice,
        roi_x_limits,
        transpose_enabled,
        background,
        was_live: bool,
    ) -> None:
        self._burst_running = True
        self._resume_live_after_burst = was_live

        self._burst_thread = QThread()
        self._burst_worker = BurstWorker(
            self._driver,
            self._fitter_burst,
            n_frames,
            roi_slice,
            roi_x_limits,
            transpose=transpose_enabled,
            background=background,
        )
        self._burst_worker.moveToThread(self._burst_thread)

        self._burst_thread.started.connect(self._burst_worker.run_burst)
        self._burst_worker.progress.connect(self.burst_progress)
        self._burst_worker.finished.connect(self._handle_burst_finished)
        self._burst_worker.error.connect(self._handle_burst_error)

        # Thread Lifecycle Management:
        # 1. When worker finishes, quit the thread loop
        self._burst_worker.finished.connect(self._burst_thread.quit)
        self._burst_worker.error.connect(self._burst_thread.quit)
        
        # 2. When thread finishes, enqueue a command to clean up safely in THIS thread
        # We pass the current burst_id to ensure we don't clean up the wrong thread later
        current_id = self._burst_id
        self._burst_thread.finished.connect(
            lambda: self.enqueue(CameraCommand.BURST_COMPLETED, current_id)
        )

        self._burst_thread.start()

    def _handle_burst_finished(self, result) -> None:
        self.burst_finished.emit(result)
        if self._burst_thread is not None and self._burst_thread.isRunning():
            self._burst_thread.quit()
        # Trigger cleanup explicitly to avoid hanging bursts
        current_id = self._burst_id
        self.enqueue(CameraCommand.BURST_COMPLETED, current_id)

    def _handle_burst_error(self, msg: str) -> None:
        self.burst_error.emit(msg)
        if self._burst_thread is not None and self._burst_thread.isRunning():
            self._burst_thread.quit()
        # Trigger cleanup explicitly to avoid hanging bursts
        current_id = self._burst_id
        self.enqueue(CameraCommand.BURST_COMPLETED, current_id)

    def _cleanup_burst(self) -> None:
        """Called safely from within the run() loop via BURST_COMPLETED command."""
        # 1. Restore Live State if needed
        resume_live = self._resume_live_after_burst
        self._burst_running = False
        self._resume_live_after_burst = False
        
        # 2. Clean up objects
        if self._burst_worker is not None:
            self._burst_worker.deleteLater()
            self._burst_worker = None
        if self._burst_thread is not None:
            # Ensure it's truly stopped
            if self._burst_thread.isRunning():
                self._burst_thread.quit()
                self._burst_thread.wait()
            self._burst_thread.deleteLater()
            self._burst_thread = None

        # 3. Resume Live
        if resume_live and not self._shutdown:
            try:
                self._driver.start_stream()
                self._live_running = True
                self.live_state_changed.emit(True)
            except Exception as e:
                self.error.emit(str(e))

    def _wait_for_burst_thread(self, timeout_ms: int = 5000) -> None:
        """Safely wait for the burst thread to finish."""
        if self._burst_thread is not None:
            # Disconnect the completion signal so we don't trigger double cleanup
            try:
                self._burst_thread.finished.disconnect()
            except Exception:
                pass

            if self._burst_thread.isRunning():
                self._logger.info("Stopping burst thread...")
                self._burst_thread.quit()
                finished = self._burst_thread.wait(timeout_ms)
                if not finished:
                    self._logger.warning("Burst thread did not finish cleanly.")
            
            # Final cleanup
            self._cleanup_burst()