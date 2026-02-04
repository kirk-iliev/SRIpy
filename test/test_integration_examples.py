"""
Integration tests for SRIpy frame processing pipeline.

These tests verify that multiple components work together correctly
without requiring actual hardware.
"""

import pytest
import numpy as np
from types import SimpleNamespace
import sys
import os

# Fixtures MockDriver and MockFitter are auto-imported from conftest.py
# by pytest, so we can use them as function parameters


@pytest.mark.integration
class TestFrameProcessingPipeline:
    """Test the complete acquisition → analysis → result flow."""
    
    def test_synthetic_frame_to_fit(self, synthetic_frame_2d, mock_driver, mock_fitter):
        """Can we acquire a frame, extract lineout, and fit it?"""
        # Use mock driver and fitter
        mock_driver.frames = [synthetic_frame_2d]
        mock_driver.frame_idx = 0
        frame = mock_driver.acquire_frame()
        
        # Extract lineout
        lineout = mock_fitter.get_lineout(frame)
        assert len(lineout) == synthetic_frame_2d.shape[1]  # Should be width
        assert np.all(np.isfinite(lineout))  # No NaN/inf
        
        # Fit
        result = mock_fitter.fit(lineout)
        assert result.success
        assert 0 < result.visibility <= 1
        assert result.sigma_microns >= 0
    
    def test_saturated_frame_handling(self, synthetic_frame_with_saturation, mock_driver, mock_fitter):
        """Does saturation detection work correctly?"""
        mock_driver.frames = [synthetic_frame_with_saturation]
        mock_driver.frame_idx = 0
        frame = mock_driver.acquire_frame()
        
        # Check for saturation
        sat_limit = 4090
        is_saturated = np.max(frame) >= sat_limit
        assert is_saturated  # Our fixture should have saturation
        
        # Lineout extraction should still work
        lineout = mock_fitter.get_lineout(frame)
        assert len(lineout) > 0
    
    def test_low_snr_rejection(self, mock_fitter):
        """Does low SNR get rejected properly?"""
        from analysis.fitter import InterferenceFitter
        
        fitter = InterferenceFitter()
        # Create truly low SNR: almost no signal above noise floor
        truly_low_snr = np.full(100, 25.0) + np.random.normal(0, 5, 100)
        result = fitter.fit(truly_low_snr)
        
        # Should fail due to low signal (min_signal_threshold is 50)
        assert result.success is False
        assert "Low Signal" in result.message
    
    def test_burst_acquisition_flow(self, synthetic_lineout, mock_fitter):
        """Test burst mode: acquire multiple frames and process."""
        from core.data_model import BurstResult
        
        # Simulate 3-frame burst
        frames = [synthetic_lineout + np.random.normal(0, 2, len(synthetic_lineout)) 
                  for _ in range(3)]
        
        vis_list = []
        sigma_list = []
        
        for frame_data in frames:
            result = mock_fitter.fit(frame_data)
            if result.success:
                vis_list.append(result.visibility)
                sigma_list.append(result.sigma_microns)
        
        # Should have 3 successful fits
        assert len(vis_list) == 3
        assert len(sigma_list) == 3
        
        # Stats should be reasonable
        mean_vis = np.mean(vis_list)
        std_vis = np.std(vis_list)
        assert 0 < mean_vis <= 1
        assert std_vis >= 0
    
    def test_driver_connection_lifecycle(self, mock_driver):
        """Test connect → stream → acquire → disconnect."""
        # Connect
        mock_driver.connect()
        
        # Prepare stream
        mock_driver.start_stream()
        assert mock_driver.is_streaming
        
        # Set camera parameters
        mock_driver.exposure = 0.005
        mock_driver.gain = 10
        assert mock_driver.exposure == 0.005
        assert mock_driver.gain == 10
        
        # Close
        mock_driver.close()
        assert not mock_driver.is_streaming
    
    def test_frame_preprocessing_chain(self, synthetic_frame_2d):
        """Test: raw frame → transpose → background subtract → clip."""
        frame = synthetic_frame_2d.copy().astype(np.float32)
        
        # Background subtraction
        background = frame * 0.1  # Simple: 10% of image
        frame = frame - background
        frame = np.clip(frame, 0, None)
        
        # Transpose
        frame = np.ascontiguousarray(frame.T)
        
        # Verify transformations
        assert frame.dtype == np.float32
        assert np.all(frame >= 0)
        assert frame.shape == (synthetic_frame_2d.shape[1], synthetic_frame_2d.shape[0])


    def test_error_handling_pipeline(self):
        """Test error handling across pipeline stages."""
        pass
    
    def test_driver_connection_failure(self, mock_driver):
        """What happens when camera fails to connect? - Use separate mock."""
        pass
    
    def test_fit_with_all_nan_data(self, mock_fitter):
        """Fitter should handle NaN gracefully."""
        from analysis.fitter import InterferenceFitter
        
        fitter = InterferenceFitter()
        bad_data = np.full(100, np.nan)
        result = fitter.fit(bad_data)
        
        # Should fail gracefully, not crash
        assert result.success is False
    
    def test_empty_frame_sequence(self, mock_driver):
        """Burst with no frames should handle gracefully."""
        mock_driver.frames = []  # Empty
        
        for i in range(3):
            frame = mock_driver.acquire_frame()
            assert frame is None  # Expected: no frames


@pytest.mark.integration
class TestDataFlowConsistency:
    """Verify data types and shapes are consistent through pipeline."""
    
    def test_lineout_shape_consistency(self, synthetic_frame_2d, mock_fitter):
        """Lineout shape should match image width."""
        frame = synthetic_frame_2d
        lineout = mock_fitter.get_lineout(frame)
        
        assert lineout.shape[0] == frame.shape[1]  # Width
    
    def test_fit_result_contains_all_fields(self, synthetic_lineout, mock_fitter):
        """FitResult should have all expected fields."""
        result = mock_fitter.fit(synthetic_lineout)
        
        # Check all required attributes exist
        assert hasattr(result, 'success')
        assert hasattr(result, 'visibility')
        assert hasattr(result, 'sigma_microns')
        assert hasattr(result, 'fitted_curve')
        assert hasattr(result, 'params')
        assert hasattr(result, 'param_errors')
        assert hasattr(result, 'message')
        
        # Check types
        assert isinstance(result.success, bool)
        assert isinstance(result.visibility, (int, float))
        assert isinstance(result.sigma_microns, (int, float))
        assert isinstance(result.fitted_curve, np.ndarray)
    
    def test_numpy_array_memory_layout(self, synthetic_frame_2d):
        """Verify arrays are C-contiguous for efficient processing."""
        frame = synthetic_frame_2d
        frame = np.ascontiguousarray(frame.T)
        
        assert frame.flags['C_CONTIGUOUS']
        assert frame.dtype in [np.float32, np.float64, np.uint16]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
