import logging
import numpy as np
import unittest
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from analysis.fitter import InterferenceFitter

logger = logging.getLogger(__name__)


class TestInterferencePhysics(unittest.TestCase):
    def setUp(self):
        # Configure standard physics parameters (same as your config)
        self.fitter = InterferenceFitter(wavelength=550e-9, slit_separation=0.05, distance=0.3)

    def generate_synthetic_beam(self, target_sigma_microns, noise_level=0.0):
        """Generates a 1D lineout mimicking a real synchrotron beam."""
        x = np.arange(0, 1200) # Pixel space
        
        # 1. Reverse-calculate Visibility from Target Sigma
        # Sigma = C * sqrt(0.5 * ln(1/V))  -->  V = exp( -2 * (Sigma/C)^2 )
        coeff = (self.fitter.wavelength * self.fitter.distance) / (np.pi * self.fitter.slit_sep)
        target_sigma = target_sigma_microns * 1e-6
        expected_visibility = np.exp(-2 * (target_sigma / coeff)**2)
        
        # 2. Generate the ideal curve using your own model
        # Params: [baseline, amp, sinc_w, sinc_x0, visibility, sine_k, sine_phase]
        params = [
            100.0,              # Baseline
            3000.0,             # Amplitude
            0.005,              # Sinc Width (Envelope)
            600.0,              # Center Pixel
            expected_visibility,# Visibility (derived from sigma)
            0.35,               # Fringe Frequency
            0.0                 # Phase
        ]
        
        y_clean = self.fitter._full_interference_model(x, *params)
        
        # 3. Add Gaussian Read Noise (Simulate Camera)
        noise = np.random.normal(0, noise_level, size=x.shape)
        return x, y_clean + noise, expected_visibility

    def test_sigma_recovery(self):
        """Can we recover 45um Sigma from a noisy signal?"""
        logger.info("--- Running Physics Recovery Test ---")
        
        # Generate a beam that equals exactly 45.0 microns
        x, y_data, true_vis = self.generate_synthetic_beam(target_sigma_microns=1.6, noise_level=50.0)
        
        # Run the fit
        result = self.fitter.fit(y_data)
        
        logger.info(f"Target Sigma: 1.60 um")
        logger.info(f"Fitted Sigma: {result.sigma_microns:.2f} um")
        logger.info(f"Visibility:   {result.visibility:.4f} (True: {true_vis:.4f})")

        # Assertions
        self.assertTrue(result.success, "Fitter failed to converge on valid data")
        self.assertAlmostEqual(result.sigma_microns, 1.6, delta=0.2, 
                             msg="Fitter result deviated by more than 0.2 microns!")

if __name__ == '__main__':
    unittest.main()