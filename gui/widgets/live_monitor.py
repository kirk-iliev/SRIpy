from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
from typing import Tuple
import numpy as np

try:
    import matplotlib
except Exception:  # matplotlib is optional at runtime
    matplotlib = None

class LiveMonitorWidget(QWidget):
    # Signals to communicate with Main Window
    roi_changed = pyqtSignal()
    roi_drag_start = pyqtSignal()   # Emitted when user is actively dragging
    roi_drag_end = pyqtSignal()     # Emitted when user releases the mouse

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self._last_img_shape = None
        self._last_lineout_range = None

        # Camera Frame Plot
        self.image_container = pg.GraphicsLayoutWidget()
        self.image_plot = self.image_container.addPlot(title="Camera Frame")  # type: ignore
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)

        # Use matplotlib colormaps if available
        if matplotlib is not None:
            try:
                cmap = matplotlib.colormaps['jet']
            except (AttributeError, KeyError):
                # Fallback for older matplotlib
                from matplotlib import cm
                cmap = cm.get_cmap('jet')

            if cmap is not None:
                lut = cmap(np.linspace(0, 1, 256))[:, :3] * 255
                self.image_item.setLookupTable(np.array(lut, dtype=np.uint8))

        # ROI: Vertical Binning Region (Rows)
        self.roi_rows = pg.LinearRegionItem(orientation='horizontal', brush=(0, 50, 255, 50))  # type: ignore
        self.roi_rows.setRegion([400, 800])
        # Bounds will be set dynamically in update_image
        self.image_plot.addItem(self.roi_rows)

        # Connect Vertical ROI signals
        self.roi_rows.sigRegionChanged.connect(self._handle_region_change)
        self.roi_rows.sigRegionChangeFinished.connect(lambda: self.roi_drag_end.emit())

        # Lineout Plot
        self.lineout_plot = pg.PlotWidget(title="Interference Profile")
        self.lineout_plot.showGrid(x=True, y=True)
        self.curve_raw = self.lineout_plot.plot(pen=pg.mkPen('w', width=2), name="Raw")
        self.curve_fit = self.lineout_plot.plot(pen=pg.mkPen('r', width=3, style=Qt.PenStyle.DashLine), name="Fit")

        # ROI: Fit Width Region (Horizontal)
        self.roi_fit_width = pg.LinearRegionItem(orientation='vertical', brush=(0, 255, 0, 30))  # type: ignore
        self.roi_fit_width.setRegion([800, 1200])
        # Bounds will be set dynamically in update_image
        self.lineout_plot.addItem(self.roi_fit_width)

        # Connect Horizontal ROI signals
        self.roi_fit_width.sigRegionChanged.connect(self._handle_region_change)
        self.roi_fit_width.sigRegionChangeFinished.connect(lambda: self.roi_drag_end.emit())

        layout.addWidget(self.image_container, stretch=3)
        layout.addWidget(self.lineout_plot, stretch=2)

    def _handle_region_change(self):
        """Internal helper to emit both changed and drag_start signals."""
        self.roi_drag_start.emit()
        self.roi_changed.emit()

    def update_image(self, img_data):
        if img_data is None:
            return

        # Ensure view auto-ranges when dimensions change (e.g., transpose)
        if self._last_img_shape != img_data.shape:
            self._last_img_shape = img_data.shape
            self.image_plot.enableAutoRange()
            self.image_plot.setAspectLocked(False)

        self.image_item.setImage(img_data, autoLevels=True, autoRange=True)

        # Update bounds dynamically based on incoming image size
        h, w = img_data.shape
        self.roi_rows.setBounds([0, h])
        self.roi_fit_width.setBounds([0, w])

    def update_lineout(self, x_data, y_data):
        self.curve_raw.setData(x_data, y_data)
        self._auto_range_lineout(y_data)

    def update_fit(self, x_data, y_data):
        self.curve_fit.setData(x_data, y_data)
        self._auto_range_lineout(y_data)

    def _auto_range_lineout(self, y_data):
        if y_data is None:
            return
        try:
            y_arr = np.asarray(y_data)
            if y_arr.size == 0 or not np.any(np.isfinite(y_arr)):
                return
            y_min = float(np.nanmin(y_arr))
            y_max = float(np.nanmax(y_arr))
            current_range = y_max - y_min
            if current_range <= 0:
                return

            # Auto-range if range changes significantly (e.g., switching live/static)
            if self._last_lineout_range is None:
                should_autorange = True
            else:
                ratio = current_range / max(self._last_lineout_range, 1e-9)
                should_autorange = ratio > 5.0 or ratio < 0.2

            if should_autorange:
                self.lineout_plot.enableAutoRange(axis='xy', enable=True)
                self.lineout_plot.autoRange()
                self._last_lineout_range = current_range
        except Exception:
            return

    def get_roi_rows(self) -> Tuple[float, float]:
        return self.roi_rows.getRegion()  # type: ignore

    def get_roi_width(self) -> Tuple[float, float]:
        return self.roi_fit_width.getRegion()  # type: ignore

    def set_roi_rows(self, min_val, max_val):
        self.roi_rows.setRegion([min_val, max_val])

    def set_roi_width(self, min_val, max_val):
        self.roi_fit_width.setRegion([min_val, max_val])