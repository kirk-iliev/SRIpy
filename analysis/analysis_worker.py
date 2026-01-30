from PyQt6.QtCore import QObject, pyqtSignal

class AnalysisWorker(QObject):
    result_ready = pyqtSignal(object, object)

    def __init__(self, fitter):
        super().__init__()
        self.fitter = fitter

    def process_fit(self, y_data, x_data):
        try:
            fit_result = self.fitter.fit(y_data)
            self.result_ready.emit(fit_result, x_data)
        except Exception as e:
            from analysis.fitter import FitResult
            failed_res = FitResult(success=False, message=str(e))
            self.result_ready.emit(failed_res, x_data)