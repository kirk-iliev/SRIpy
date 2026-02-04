import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class FitResult:
    """Standardized container for interference fit results."""
    success: bool
    visibility: float = 0.0
    sigma_microns: float = 0.0
    fitted_curve: Optional[np.ndarray] = None
    params: Optional[Dict[str, float]] = None
    param_errors: Optional[Dict[str, float]] = None
    pcov: Optional[np.ndarray] = None
    message: str = ""

class InterferenceFitter:
    def __init__(self, wavelength: float = 550e-9, slit_separation: float = 0.05, distance: float = 16.5, min_signal: float = 50.0):
        import logging
        self.logger = logging.getLogger(__name__)
        self.wavelength = wavelength
        self.slit_sep = slit_separation
        self.distance = distance
        self.min_signal = min_signal

    def get_lineout(self, image: np.ndarray, roi_slice: Optional[slice] = None) -> np.ndarray:
        if roi_slice:
            data = image[roi_slice, :] if image.ndim == 2 else image
        else:
            data = image

        if data.ndim > 1:
            return np.sum(data, axis=0)
        return data

    def _gaussian(self, x: np.ndarray, baseline: float, amp: float, center: float, width: float) -> np.ndarray:
        return baseline + amp * np.exp(-((x - center) ** 2) / (2 * width ** 2))

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
        # Sanitize Input
        # Ensure we accept both numpy arrays and python lists by coercing to ndarray
        y = np.nan_to_num(np.asarray(lineout, dtype=float))
        x = np.arange(len(y))
        
        # Fail early if signal is dead (read noise only)
        if (np.max(y) - np.min(y)) < self.min_signal:
             return FitResult(success=False, message="Low Signal")
        
        # --- Gaussian Estimate (Find Center) ---
        peak_idx = np.argmax(y)
        peak_val = y[peak_idx]
        min_val = np.min(y)
        
        # Rough width estimate
        half_max = (peak_val + min_val) / 2
        above_half = np.where(y > half_max)[0]
        if len(above_half) > 1:
            est_width = (above_half[-1] - above_half[0]) / 2.0
        else:
            est_width = 50.0
            
        p0_gauss = [min_val, peak_val - min_val, peak_idx, est_width]
        
        try:
            # We constrain width > 0 to prevent flip errors
            # Bounds: [base, amp, center, width]
            bounds_g_min = [-np.inf, 0, 0, 0.1]
            bounds_g_max = [np.inf, np.inf, len(y), len(y)]
            
            popt_g, _ = curve_fit(self._gaussian, x, y, p0=p0_gauss, 
                      bounds=(bounds_g_min, bounds_g_max), maxfev=1000)
            center_guess = popt_g[2]
        except RuntimeError as e:
            # Fit failed to converge; fall back to peak position
            self.logger.debug(f"Gaussian fit failed to converge: {e}")
            center_guess = peak_idx
        except ValueError as e:
            # Invalid bounds or parameters
            self.logger.debug(f"Gaussian fit invalid parameters: {e}")
            center_guess = peak_idx
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.warning(f"Gaussian fit unexpected error: {type(e).__name__}: {e}")
            center_guess = peak_idx

        # FFT (Find Frequency) 
        # MATLAB does this: ilo=10; ihi=min(200, length(s)); ffts=abs(fft(s));
        try:
            y_centered = y - np.mean(y)
            # Windowing helps reduce edge artifacts in FFT
            window = np.hanning(len(y))
            yf = np.fft.rfft(y_centered * window)
            xf = np.fft.rfftfreq(len(y))
            
            # Ignore DC component and very low freq
            fft_mag = np.abs(yf)
            fft_mag[0:5] = 0 
            
            dominant_idx = np.argmax(fft_mag)
            est_freq = xf[dominant_idx]
            est_sine_k = 2 * np.pi * est_freq
        except Exception as e:
            # FFT calculation failed; use fallback frequency estimate
            self.logger.debug(f"FFT frequency estimation failed: {type(e).__name__}: {e}")
            est_sine_k = 0.3  # Fallback empirical value

        # Params: [baseline, amp, sinc_w, sinc_x0, visibility, sine_k, sine_phase]
        
        # Initial Guesses
        est_base = min_val
        est_amp = peak_val - min_val
        est_sinc_w = 2.0 / (est_width * 2) # Sinc width is roughly related to Gaussian width
        
        p0_final = [est_base, est_amp, est_sinc_w, center_guess, 0.5, est_sine_k, 0.0]
        
        # 1. Visibility must be [0, 1]
        # 2. Sinc width must be positive
        # 3. Frequency (sine_k) constrained to +/- 20% of FFT estimate to prevent locking on noise
        freq_tol = 0.2 * est_sine_k
        k_min = max(0, est_sine_k - freq_tol)
        k_max = est_sine_k + freq_tol
        
        bounds_min = [-np.inf, 0, 0, 0, 0.0, k_min, -np.pi]
        bounds_max = [np.inf, np.inf, 1.0, len(y), 1.0, k_max, np.pi]
        
        try:
            popt, pcov = curve_fit(
                self._full_interference_model, x, y, 
                p0=p0_final, bounds=(bounds_min, bounds_max), maxfev=5000
            )
            
            vis = popt[4]
            sigma = self.calculate_sigma(vis)

            # Full parameter dictionary (named clearly)
            params = {
                'baseline': float(popt[0]),
                'amplitude': float(popt[1]),
                'sinc_width': float(popt[2]),
                'sinc_center': float(popt[3]),
                'visibility': float(popt[4]),
                'sine_k': float(popt[5]),
                'sine_phase': float(popt[6]),
            }

            # Try to compute standard errors from the covariance matrix
            param_errors = None
            try:
                perr = np.sqrt(np.abs(np.diag(pcov)))
                param_errors = {
                    'baseline': float(perr[0]),
                    'amplitude': float(perr[1]),
                    'sinc_width': float(perr[2]),
                    'sinc_center': float(perr[3]),
                    'visibility': float(perr[4]),
                    'sine_k': float(perr[5]),
                    'sine_phase': float(perr[6]),
                }
            except Exception:
                param_errors = None

            return FitResult(
                success=True,
                visibility=vis,
                sigma_microns=sigma * 1e6, # Convert to microns for display
                fitted_curve=self._full_interference_model(x, *popt),
                params=params,
                param_errors=param_errors,
                pcov=pcov
            )
            
        except Exception as e:
            return FitResult(success=False, message=f"Fit Failed: {str(e)}")
                

    def calculate_sigma(self, visibility: float) -> float:
        # Validate input: reject NaN, inf, and out-of-range values
        if not np.isfinite(visibility):
            self.logger.warning(f"calculate_sigma: received non-finite visibility: {visibility}")
            return 0.0
        if visibility <= 0.001 or visibility >= 0.999:
            return 0.0
        coeff = (self.wavelength * self.distance) / (np.pi * self.slit_sep)
        try:
            return coeff * np.sqrt(0.5 * np.log(1.0 / visibility))
        except ValueError as e:
            self.logger.warning(f"calculate_sigma: math domain error with visibility={visibility}: {e}")
            return 0.0