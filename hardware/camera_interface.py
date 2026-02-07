from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class CameraInterface(ABC):
    """ Abstract base class for camera interfaces """

    @abstractmethod
    def connect(self):
        """Initialize camera connection"""
        raise NotImplementedError

    @property
    @abstractmethod
    def exposure(self) -> float:
        """Get camera exposure time in seconds"""
        raise NotImplementedError

    @exposure.setter
    @abstractmethod
    def exposure(self, exposure_time_s: float):
        """Set camera exposure time in seconds"""
        raise NotImplementedError
        
    @property
    @abstractmethod
    def gain(self) -> float:
        """Get camera gain in decibels"""
        raise NotImplementedError

    @gain.setter
    @abstractmethod
    def gain(self, gain_db: float):
        """Set camera gain in decibels"""
        raise NotImplementedError

    @abstractmethod
    def acquire_frame(self, timeout: float = 3.0) -> Optional[np.ndarray]:
        """Acquire a single frame from the camera. Returns None on timeout."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close camera connection and release resources safely"""
        raise NotImplementedError
    
