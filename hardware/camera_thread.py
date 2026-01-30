from PyQt6.QtCore import pyqtSignal, QObject, QThread
class CameraWorker (QObject):

    frame_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, driver):
        super().__init__()
        self.driver = driver
        self._is_running = False
    def start_acquire(self):
        try:
            self._is_running = True
            self.driver.start_stream()

            while self._is_running:
                frame = self.driver.acquire_frame(timeout=0.5)
                if frame is not None:
                    self.frame_ready.emit(frame)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.driver.stop_stream()

    def stop_acquire(self):
        self._is_running = False
        