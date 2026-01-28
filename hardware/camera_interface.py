from abc import ABC, abstractmethod
import numpy as np

class CameraInterface(ABC):
    """ Abstract base class for camera interfaces """

    @abstractmethod
    def connect(self):
        """Initialize camera connection"""
        pass

    @abstractmethod
    def set_exposure(self, exposure_time_s: float):
        """Set camera exposure time in seconds"""
        pass

    @abstractmethod
    def set_gain(self, gain_db: float):
        """Set camera gain in decibels"""
        pass

    @abstractmethod
    def acquire_frame(self) -> np.ndarray:
        """Acquire a single frame from the camera
        Returns: 2D numpy array"""
        pass

    @abstractmethod
    def close(self):
        """Close camera connection and release resources safely"""
        pass

    