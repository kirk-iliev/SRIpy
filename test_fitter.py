import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.fitter import InterferenceFitter

class TestInterferenceFitter(unittest.TestCase):
    def setUp(self):
        self.wavelength = 550e-9
        self.slit_sep = 0.05      # 50 mm = 0.05 m
        self.distance = 16.5      # 16.5 m
        self.fitter = InterferenceFitter(
            wavelength=self.wavelength, 
            slit_separation=self.slit_sep, 
            distance=self.distance
        )

    def generate_synthetic_data(self, visibility, sigma_microns, noise_level=0.0):
        """
        Generates a synthetic interference pattern based on physical laws.
        Returns: (x_data, y_data)
        """
        x = np.arange(0, 1200) # Standard ROI width
        
        # Sigma = Coeff * sqrt(0.5 * ln(1/V))
        # This allows us to double-check the math parity
        coeff = (self.wavelength * self.distance) / (np.pi * self.slit_sep)
        
        # Width roughly 1000 pixels, centered at 600
        center = 600
        width = 400.0 
        x_shifted = x - center
        val = (2.0/width) * x_shifted
        val = np.where(val == 0, 1e-10, val)
        sinc_sq = (np.sin(val) / val)**2
        
        # Fringe frequency depends on slit separation
        freq = 0.15 # Arbitrary reasonable frequency for pixels
        phase = 0.0
        modulation = 1 + visibility * np.cos(freq * x + phase)
        
        # Combine
        amplitude = 3000
        baseline = 100
        y_perfect = baseline + amplitude * sinc_sq * modulation
        
        # Add Noise
        noise = np.random.normal(0, noise_level, len(x))
        y_noisy = y_perfect + noise
        
        return x, y_noisy

    def test_perfect_data(self):
        """Test with 0 noise to verify equation correctness."""
        target_vis = 0.6
        # Sigma doesn't matter here, we just want to recover Vis
        x, y = self.generate_synthetic_data(target_vis, 100, noise_level=0.0)
        
        res = self.fitter.fit(y)
        
        print(f"\n[Perfect Data] Target Vis: {target_vis}, Fitted Vis: {res.visibility:.4f}")
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.visibility, target_vis, places=2)

    def test_noisy_data(self):
        """Test with realistic camera noise (approx 20 counts)."""
        target_vis = 0.45
        noise = 20.0 
        x, y = self.generate_synthetic_data(target_vis, 100, noise_level=noise)
        
        res = self.fitter.fit(y)
        
        print(f"[Noisy Data]   Target Vis: {target_vis}, Fitted Vis: {res.visibility:.4f}")
        self.assertTrue(res.success)
        # Allow 5% error margin for noise
        self.assertTrue(abs(res.visibility - target_vis) < 0.05)

    def test_low_visibility(self):
        """Test limit of detection (Vis < 0.1)."""
        target_vis = 0.08
        x, y = self.generate_synthetic_data(target_vis, 100, noise_level=5.0)
        
        res = self.fitter.fit(y)
        print(f"[Low Vis]      Target Vis: {target_vis}, Fitted Vis: {res.visibility:.4f}")
        self.assertTrue(abs(res.visibility - target_vis) < 0.03)

if __name__ == '__main__':
    unittest.main()