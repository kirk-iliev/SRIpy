import logging
import numpy as np
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

        x_data may be:
          - a tuple (roi_start, roi_stop): passed as roi_hint to the fitter
          - an ndarray: legacy x coordinates (backward compat)
        """
        try:
            # Determine if x_data is a ROI hint (tuple) or legacy x coords (ndarray)
            roi_hint = x_data if isinstance(x_data, tuple) else None

            # Acquire lock to ensure physics parameters are stable during fit
            with self._fitter_lock:
                fit_result = self.fitter.fit(y_data, roi_hint=roi_hint)

            # Use fit_x from result if available (new centered fitting),
            # otherwise fall back to legacy x_data
            if fit_result.fit_x is not None:
                out_x = fit_result.fit_x
            elif isinstance(x_data, np.ndarray):
                out_x = x_data
            else:
                out_x = np.arange(len(y_data))

            self.result_ready.emit(fit_result, out_x, req_id)

        except Exception as e:
            self.logger.error(f"Analysis Failed (Req {req_id}): {e}", exc_info=True)
            failed_res = FitResult(success=False, message=str(e))
            self.result_ready.emit(failed_res, x_data, req_id)