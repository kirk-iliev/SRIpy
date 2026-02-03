import sys
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from gui.main_window import InterferometerGUI

def main():
    """
    Application Entry Point.
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    sys.excepthook = handle_exception

    try:
        window = InterferometerGUI()
    except Exception:
        handle_exception(*sys.exc_info())
        sys.exit(1)

    try:
        window.show()
        sys.exit(app.exec())
    except Exception:
        handle_exception(*sys.exc_info())
        sys.exit(1)

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Catches unhandled exceptions and displays a critical error message.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print("Uncaught exception:", error_msg)
    
    if QApplication.instance():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("An unexpected error occurred.")
        msg.setInformativeText(str(exc_value))
        msg.setDetailedText(error_msg)
        msg.setWindowTitle("Critical Error")
        msg.exec()

if __name__ == "__main__":
    main()