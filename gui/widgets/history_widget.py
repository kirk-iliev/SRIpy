from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

class HistoryWidget(QWidget):
    def __init__(self, history_len=100):
        super().__init__()
        layout = QVBoxLayout(self)
        
        self.history_len = history_len
        self.history_data = np.zeros(self.history_len)
        
        self.plot_widget = pg.PlotWidget(title="Beam Size History")
        self.plot_widget.setLabel('left', 'Sigma (um)')
        self.plot_widget.setLabel('bottom', 'Frame History')
        self.plot_widget.showGrid(x=True, y=True)
        self.curve = self.plot_widget.plot(pen=pg.mkPen('c', width=2), symbol='o', symbolSize=5)
        
        layout.addWidget(self.plot_widget)

    def add_point(self, value):
        """Add a new sigma value and scroll the plot."""
        self.history_data = np.roll(self.history_data, -1)
        self.history_data[-1] = value
        self.curve.setData(self.history_data)