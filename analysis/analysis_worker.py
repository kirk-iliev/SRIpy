import logging
from PyQt6.QtCore import QObject, pyqtSignal
from analysis.fitter import FitResult

class AnalysisWorker(QObject):
    result_ready = pyqtSignal(object, object, int)

    def __init__(self, fitter, fitter_lock):
        """
        Worker thread for running heavy fitting analysis.

        Args:
            fitter: Instance of InterferenceFitter
            fitter_lock: Threading lock (RLock) shared with AcquisitionManager.
        """
        super().__init__()
        self.fitter = fitter

        # Enforce thread safety by rejecting initialization without a lock
        if fitter_lock is None:
            raise ValueError("AnalysisWorker requires a valid thread lock to protect shared physics state.")

        self._fitter_lock = fitter_lock
        self.logger = logging.getLogger(__name__)

    def process_fit(self, y_data, x_data, req_id: int):
        """
        Runs the fitting analysis in a separate thread. Acquires a lock to ensure
        that physics parameters (lambda/slit/dist) are not changed mid-fit.
        """
        try:
            # Acquire lock to ensure physics parameters are stable during fit
            with self._fitter_lock:
                fit_result = self.fitter.fit(y_data)

            self.result_ready.emit(fit_result, x_data, req_id)

        except Exception as e:
            self.logger.error(f"Analysis Failed (Req {req_id}): {e}", exc_info=True)
            failed_res = FitResult(success=False, message=str(e))
            self.result_ready.emit(failed_res, x_data, req_id)