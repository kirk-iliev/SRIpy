import logging
from PyQt6.QtCore import QObject, pyqtSignal
from analysis.fitter import FitResult

class AnalysisWorker(QObject):
    result_ready = pyqtSignal(object, object, int)

    def __init__(self, fitter, fitter_lock=None):
        super().__init__()
        self.fitter = fitter
        self._fitter_lock = fitter_lock
        self.logger = logging.getLogger(__name__)

    def process_fit(self, y_data, x_data, req_id: int):
        
        try:
            if self._fitter_lock is None:
                fit_result = self.fitter.fit(y_data)
            else:
                with self._fitter_lock:
                    fit_result = self.fitter.fit(y_data)
            self.result_ready.emit(fit_result, x_data, req_id)
            
        except Exception as e:
            self.logger.error(f"Analysis Failed: {e}", exc_info=True)
            failed_res = FitResult(success=False, message=str(e))
            self.result_ready.emit(failed_res, x_data, req_id)