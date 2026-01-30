from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

class LiveMonitorWidget(QWidget):
    # Signals to communicate with Main Window
    roi_changed = pyqtSignal()      
    roi_drag_start = pyqtSignal()   # Emitted when user is actively dragging
    roi_drag_end = pyqtSignal()     # Emitted when user releases the mouse

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Camera Frame Plot
        self.image_container = pg.GraphicsLayoutWidget()
        self.image_plot = self.image_container.addPlot(title="Camera Frame")
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)
        
        # ROI: Vertical Binning Region
        self.roi_rows = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Horizontal, brush=(0, 50, 255, 50))
        self.roi_rows.setRegion([400, 800]) 
        self.image_plot.addItem(self.roi_rows)
        
        # Lineout Plot
        self.lineout_plot = pg.PlotWidget(title="Interference Profile")
        self.lineout_plot.showGrid(x=True, y=True)
        self.curve_raw = self.lineout_plot.plot(pen=pg.mkPen('w', width=2), name="Raw")
        self.curve_fit = self.lineout_plot.plot(pen=pg.mkPen('r', width=3, style=Qt.PenStyle.DashLine), name="Fit")
        
        # ROI: Fit Width Region
        self.roi_fit_width = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical, brush=(0, 255, 0, 30))
        self.roi_fit_width.setRegion([800, 1200]) 
        self.lineout_plot.addItem(self.roi_fit_width)

        # sigRegionChanged fires *during* the drag. 
        # We use this to trigger 'roi_drag_start' which sets user_is_interacting=True
        self.roi_fit_width.sigRegionChanged.connect(self._handle_region_change)
        
        # sigRegionChangeFinished fires on *release*.
        # This triggers 'roi_drag_end' which sets user_is_interacting=False
        self.roi_fit_width.sigRegionChangeFinished.connect(self.roi_drag_end)
        
        layout.addWidget(self.image_container, stretch=3)
        layout.addWidget(self.lineout_plot, stretch=2)

    def _handle_region_change(self):
        """Internal helper to emit both changed and drag_start signals."""
        self.roi_drag_start.emit()
        self.roi_changed.emit()

    def update_image(self, img_data):
        self.image_item.setImage(img_data, autoLevels=False, levels=(0, 4095))

    def update_lineout(self, x_data, y_data):
        self.curve_raw.setData(y_data)

    def update_fit(self, x_data, y_data):
        self.curve_fit.setData(x_data, y_data)
        
    def get_roi_rows(self):
        return self.roi_rows.getRegion()

    def get_roi_width(self):
        return self.roi_fit_width.getRegion()

    def set_roi_rows(self, min_val, max_val):
        self.roi_rows.setRegion([min_val, max_val])

    def set_roi_width(self, min_val, max_val):
        self.roi_fit_width.setRegion([min_val, max_val])