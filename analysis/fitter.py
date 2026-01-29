import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import warnings 

@dataclass
class FitResult:
    """Standardized container for interference fit results."""
    success: bool
    visibility: float = 0.0
    sigma_microns: float = 0.0
    fitted_curve: Optional[np.ndarray] = None
    params: Optional[Dict[str, float]] = None
    message: str = ""

class InterferenceFitter:
    def __init__(self, wavelength: float = 550e-9, slit_separation: float = 0.05, distance: float = 16.5):
        self.wavelength = wavelength
        self.slit_sep = slit_separation
        self.distance = distance

    def get_lineout(self, image: np.ndarray, roi_slice: slice = None) -> np.ndarray:
        if roi_slice:
            data = image[roi_slice, :] if image.ndim == 2 else image
        else:
            data = image

        if data.ndim > 1:
            return np.sum(data, axis=0)
        return data

    def _gaussian(self, x: np.ndarray, baseline: float, amp: float, center: float, width: float) -> np.ndarray:
        return baseline + amp * np.exp(-((x - center) ** 2) / (width ** 2))

    def _sinc_sq_envelope(self, x: np.ndarray, baseline: float, amp: float, width: float, center: float) -> np.ndarray:
        x_shifted = x - center
        val = width * x_shifted
        val = np.where(np.isclose(val, 0), 1e-10, val)
        sinc_term = np.sin(val) / val
        return baseline + amp * (sinc_term ** 2)

    def _full_interference_model(self, x: np.ndarray, baseline: float, amp: float, sinc_w: float, 
                                 sinc_x0: float, visibility: float, sine_freq: float, sine_phase: float) -> np.ndarray:
        val = sinc_w * (x - sinc_x0)
        val = np.where(np.isclose(val, 0), 1e-10, val)
        sinc_sq = amp * ((np.sin(val) / val) ** 2)
        interf = 1 + visibility * np.sin(sine_freq * x + sine_phase)
        return baseline + sinc_sq * interf

    def fit(self, lineout: np.ndarray) -> FitResult:
        # 1. Sanitize Input
        y = np.nan_to_num(lineout.astype(float))
        x = np.arange(len(y))
        
        # If signal is too weak (read-noise only), fail early.
        if (np.max(y) - np.min(y)) < 50:
             return FitResult(success=False, message="Low Signal")
        
        # --- 1. Gaussian Estimate (Center) ---
        peak_idx = np.argmax(y)
        peak_val = y[peak_idx]
        min_val = np.min(y)
        
        half_max = (peak_val + min_val) / 2
        above_half = np.where(y > half_max)[0]
        if len(above_half) > 1:
            est_width = (above_half[-1] - above_half[0]) / 2.0
        else:
            est_width = 50.0
            
        p0_gauss = [min_val, peak_val - min_val, peak_idx, est_width]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                popt_g, _ = curve_fit(self._gaussian, x, y, p0=p0_gauss, maxfev=2000)
                if not np.all(np.isfinite(popt_g)): raise ValueError("Gaussian Fit NaN")
                
                center_guess = popt_g[2]
                width_guess = abs(popt_g[3])
                if width_guess < 1 or width_guess > len(y): center_guess = peak_idx
                
            except Exception:
                center_guess = peak_idx
                popt_g = p0_gauss 
                
            # --- 2. Sinc Estimate (Envelope) ---
            safe_width = abs(popt_g[3]) if abs(popt_g[3]) > 1e-5 else 50.0
            p0_sinc = [min_val, (peak_val - min_val), 2.0 / safe_width, center_guess]
            
            try:
                popt_s, _ = curve_fit(self._sinc_sq_envelope, x, y, p0=p0_sinc, maxfev=2000)
                if not np.all(np.isfinite(popt_s)): raise ValueError("Sinc Fit NaN")
                est_base, est_amp, est_sinc_w, est_sinc_x0 = popt_s
            except Exception:
                est_base, est_amp = min_val, peak_val - min_val
                est_sinc_w = 2.0 / 50.0 
                est_sinc_x0 = center_guess
                popt_s = [est_base, est_amp, est_sinc_w, est_sinc_x0]

            # --- 3. FFT (Freq) ---
            try:
                y_centered = y - np.mean(y)
                window = np.hanning(len(y))
                yf = np.fft.fft(y_centered * window)
                xf = np.fft.fftfreq(len(y))
                pos_mask = xf > 0
                fft_mag = np.abs(yf[pos_mask])
                dominant_idx = np.argmax(fft_mag)
                est_sine_k = 2 * np.pi * xf[pos_mask][dominant_idx]
            except Exception:
                est_sine_k = 0.3

            # --- 4. Full Fit ---
            p0_final = [est_base, est_amp, est_sinc_w, est_sinc_x0, 0.5, est_sine_k, 0.0]
            bounds_min = [0, 0, 0, -np.inf, 0.0, 0, -np.pi]
            bounds_max = [np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, np.pi]
            
            try:
                popt, _ = curve_fit(
                    self._full_interference_model, x, y, 
                    p0=p0_final, bounds=(bounds_min, bounds_max), maxfev=5000
                )
                
                if not np.all(np.isfinite(popt)): raise ValueError("Final Fit NaN")

                vis = popt[4]
                sigma = self.calculate_sigma(vis)
                
                return FitResult(
                    success=True,
                    visibility=vis,
                    sigma_microns=sigma * 1e6,
                    fitted_curve=self._full_interference_model(x, *popt),
                    params={'baseline': popt[0], 'amplitude': popt[1], 'vis': vis}
                )
                
            except Exception as e:
                # If fitting fails (noise, occlusion, etc), simply report failure.
                # The GUI will clear the plot, avoiding the "jumping red line" effect.
                return FitResult(
                    success=False, 
                    message=f"Fit Failed: {str(e)}"
                )

    def calculate_sigma(self, visibility: float) -> float:
        if visibility <= 0.001 or visibility >= 0.999:
            return 0.0
        coeff = (self.wavelength * self.distance) / (np.pi * self.slit_sep)
        try:
            return coeff * np.sqrt(0.5 * np.log(1.0 / visibility))
        except ValueError:
            return 0.0