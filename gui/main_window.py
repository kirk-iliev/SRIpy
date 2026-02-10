import os
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QTabWidget

from gui.widgets.live_monitor import LiveMonitorWidget
from gui.widgets.history_widget import HistoryWidget
from gui.widgets.control_panel import ControlPanelWidget


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class InterferometerView(QMainWindow):
    close_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRIpy")
        self.resize(1300, 950)
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.tabs = QTabWidget()
        self.live_widget = LiveMonitorWidget()
        self.history_widget = HistoryWidget()
        self.tabs.addTab(self.live_widget, "Live Monitor")
        self.tabs.addTab(self.history_widget, "Stability History")

        self.controls = ControlPanelWidget()

        layout.addWidget(self.tabs, stretch=4)
        layout.addWidget(self.controls, stretch=1)

    def closeEvent(self, a0):
        self.close_requested.emit(a0)