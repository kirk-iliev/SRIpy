import logging
from PyQt6.QtCore import QObject, pyqtSignal
from analysis.fitter import FitResult 

class AnalysisWorker(QObject):
    result_ready = pyqtSignal(object, object)

    def __init__(self, fitter):
        super().__init__()
        self.fitter = fitter
        self.logger = logging.getLogger(__name__)

    def process_fit(self, y_data, x_data):
        # DEBUG LOG: Confirm we entered the thread
        # self.logger.debug("Worker received fit request") 
        
        try:
            # If the app hangs HERE, it's the OpenMP deadlock (Fixed by Step 1)
            fit_result = self.fitter.fit(y_data)
            self.result_ready.emit(fit_result, x_data)
            
        except Exception as e:
            self.logger.error(f"Analysis Failed: {e}", exc_info=True)
            failed_res = FitResult(success=False, message=str(e))
            self.result_ready.emit(failed_res, x_data)