"""
Test fixtures and mocks for SRIpy testing.

These are shared utilities to avoid duplicating mock objects across test files.
"""

import pytest
import numpy as np
from types import SimpleNamespace


class MockDriver:
    """Fake camera driver for testing without hardware."""
    
    def __init__(self, frames=None, raise_on_connect=False):
        """
        Args:
            frames: List of numpy arrays to return on acquire_frame() calls
            raise_on_connect: If True, raise exception on connect()
        """
        self.frames = frames or []
        self.frame_idx = 0
        self.raise_on_connect = raise_on_connect
        self._exposure = 0.005
        self._gain = 0.0
        self.is_streaming = False
    
    def connect(self):
        if self.raise_on_connect:
            raise RuntimeError("Mock: Connection failed")
    
    def acquire_frame(self, timeout=1.0):
        """Return next frame in queue, or None if depleted."""
        if self.frame_idx < len(self.frames):
            frame = self.frames[self.frame_idx]
            self.frame_idx += 1
            return frame.copy() if isinstance(frame, np.ndarray) else None
        return None
    
    def start_stream(self):
        self.is_streaming = True
    
    def stop_stream(self):
        self.is_streaming = False
    
    @property
    def exposure(self):
        return self._exposure
    
    @exposure.setter
    def exposure(self, value):
        self._exposure = value
    
    @property
    def gain(self):
        return self._gain
    
    @gain.setter
    def gain(self, value):
        self._gain = value
    
    def close(self):
        self.stop_stream()


class MockFitter:
    """Fake interference fitter for testing analysis without heavy computation."""
    
    def __init__(self, always_succeed=True):
        """
        Args:
            always_succeed: If True, all fits succeed with synthetic params
        """
        self.always_succeed = always_succeed
        self.wavelength = 550e-9
        self.slit_sep = 0.05
        self.distance = 0.3
    
    def fit(self, y_data):
        """Return synthetic fit result."""
        from analysis.fitter import FitResult
        
        if not self.always_succeed:
            return FitResult(success=False, message="Mock fit failure")
        
        return FitResult(
            success=True,
            visibility=0.5,
            sigma_microns=1.5,
            fitted_curve=np.array(y_data) * 0.9,  # Simple mock: 90% of input
            params={
                'baseline': 10.0,
                'amplitude': 100.0,
                'sinc_width': 0.02,
                'sinc_center': 512.0,
                'visibility': 0.5,
                'sine_k': 0.1,
                'sine_phase': 0.0,
            },
            param_errors=None,
            message=""
        )
    
    def get_lineout(self, image, roi_slice=None):
        """Extract 1D lineout from 2D image."""
        if roi_slice:
            data = image[roi_slice, :] if image.ndim == 2 else image
        else:
            data = image
        
        if data.ndim > 1:
            return np.sum(data, axis=0)
        return data


@pytest.fixture
def mock_driver():
    """Provide a mock camera driver."""
    return MockDriver()


@pytest.fixture
def mock_fitter():
    """Provide a mock interference fitter."""
    return MockFitter()


@pytest.fixture
def synthetic_frame_2d():
    """Provide a realistic 2D frame (1216 x 1936 pixels)."""
    h, w = 1216, 1936
    # Create interference pattern: background + sinc envelope * fringe modulation
    x = np.arange(w)
    y_coord = np.arange(h)
    
    # Simple 2D pattern: interference bands modulated by Gaussian
    xx, yy = np.meshgrid(x, y_coord)
    
    # Gaussian spatial profile
    center_x, center_y = w // 2, h // 2
    gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * 200 ** 2))
    
    # Fringe modulation (cosinusoidal)
    fringes = 1 + 0.6 * np.cos(0.1 * xx)
    
    # Combine
    intensity = 100 + 500 * gaussian * fringes
    
    # Add shot noise
    intensity += np.random.normal(0, 10, intensity.shape)
    intensity = np.clip(intensity, 0, 4095)
    
    return intensity.astype(np.uint16)


@pytest.fixture
def synthetic_lineout():
    """Provide a realistic 1D lineout (interference fringe pattern)."""
    x = np.arange(1024)
    
    # Model: baseline + envelope * modulation
    baseline = 50.0
    amplitude = 300.0
    
    # Sinc² envelope
    center = 512.0
    width = 0.02
    arg = width * (x - center)
    arg = np.where(np.isclose(arg, 0), 1e-10, arg)
    envelope = (np.sin(arg) / arg) ** 2
    
    # Cosine modulation (fringes)
    visibility = 0.6
    fringe_k = 0.08
    modulation = 1 + visibility * np.cos(fringe_k * x)
    
    # Combine
    y = baseline + amplitude * envelope * modulation
    
    # Add shot noise
    y += np.random.normal(0, 5, y.shape)
    y = np.clip(y, 0, 4095)
    
    return y.astype(np.float32)


@pytest.fixture
def synthetic_frame_with_saturation():
    """Provide a frame with some saturated pixels (intensity >= 4090)."""
    frame = np.random.uniform(0, 1000, (1216, 1936))
    # Add saturated region
    frame[600:700, 900:1000] = 4095
    return frame.astype(np.uint16)


@pytest.fixture
def low_snr_lineout():
    """Provide a noisy lineout with low signal-to-noise ratio."""
    # Mostly noise, weak signal
    noise = np.random.normal(100, 30, 1024)
    noise = np.clip(noise, 0, 4095)
    return noise.astype(np.float32)
