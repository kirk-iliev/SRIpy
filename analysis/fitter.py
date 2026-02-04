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

    def _sine_model(self, x: np.ndarray, baseline: float, amp: float, freq: float, phase: float) -> np.ndarray:
        return baseline + amp * np.sin(freq * x + phase)

    def _full_interference_model(self, x: np.ndarray, baseline: float, amp: float, sinc_w: float, 
                                 sinc_x0: float, visibility: float, sine_freq: float, sine_phase: float) -> np.ndarray:
        val = sinc_w * (x - sinc_x0)
        val = np.where(np.isclose(val, 0), 1e-10, val)
        sinc_sq = amp * ((np.sin(val) / val) ** 2)
        interf = 1 + visibility * np.sin(sine_freq * x + sine_phase)
        return baseline + sinc_sq * interf

    def fit(self, lineout: np.ndarray) -> FitResult:
        # Sanitize Input
        y = np.nan_to_num(np.asarray(lineout, dtype=float))
        x = np.arange(len(y))
        
        # Fail early if signal is dead
        if (np.max(y) - np.min(y)) < self.min_signal:
             return FitResult(success=False, message="Low Signal")
        
        # --- Stage 0: Gaussian Estimate (Find Center) ---
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
            bounds_g_min = [-np.inf, 0, 0, 0.1]
            bounds_g_max = [np.inf, np.inf, len(y), len(y)]
            
            popt_g, _ = curve_fit(self._gaussian, x, y, p0=p0_gauss, 
                      bounds=(bounds_g_min, bounds_g_max), maxfev=1000)
            center_guess = popt_g[2]
            width_guess = popt_g[3]
        except Exception:
            center_guess = peak_idx
            width_guess = est_width

        # --- Stage 1: Envelope Fit (Sinc^2) ---
        # Isolates the envelope shape to lock down center and width
        est_sinc_w = 2.0 / (width_guess * 2) 
        p0_env = [min_val, peak_val - min_val, est_sinc_w, center_guess]
        
        try:
            # Bounds: [base, amp, width, center]
            popt_env, _ = curve_fit(self._sinc_sq_envelope, x, y, p0=p0_env, maxfev=2000)
            
            # Update guesses with robust envelope results
            env_base, env_amp, env_w, env_center = popt_env
        except Exception:
            # Fallback to Gaussian results if Sinc fit fails
            env_base, env_amp, env_w, env_center = min_val, peak_val-min_val, est_sinc_w, center_guess

        # --- Stage 2: Sine Fit (Fringes) ---
        # FFT for frequency estimate
        try:
            y_centered = y - np.mean(y)
            window = np.hanning(len(y))
            yf = np.fft.rfft(y_centered * window)
            xf = np.fft.rfftfreq(len(y))
            fft_mag = np.abs(yf)
            
            # Dynamic low-frequency masking based on envelope width
            # The envelope spectrum is roughly 0 to env_w. Fringes must be > env_w.
            # k = 2*pi*f_idx/N  =>  f_idx = k*N / 2*pi
            cutoff_k = 1.5 * env_w  # Mask frequencies up to 1.5x the envelope width
            cutoff_idx = int((cutoff_k * len(y)) / (2 * np.pi))
            cutoff_idx = max(5, cutoff_idx) # Ensure we at least drop DC/very low freq
            
            fft_mag[0:cutoff_idx] = 0 
            
            dominant_idx = np.argmax(fft_mag)
            est_freq = xf[dominant_idx]
            est_sine_k = 2 * np.pi * est_freq
        except Exception:
            est_sine_k = 0.3

        # Refine sine params by fitting only the central region
        try:
            npts = 50
            c_idx = int(env_center)
            x_min, x_max = max(0, c_idx - npts), min(len(y), c_idx + npts)
            
            if x_max > x_min + 10:
                y_roi = y[x_min:x_max]
                x_roi = x[x_min:x_max]
                
                # Estimate sine amp from envelope amp in this region
                p0_sine = [np.mean(y_roi), (np.max(y_roi)-np.min(y_roi))/2, est_sine_k, 0.0]
                
                # Loose bounds on frequency (+/- 20%)
                freq_tol = 0.2 * est_sine_k
                bs_min = [-np.inf, 0, max(0, est_sine_k - freq_tol), -np.pi]
                bs_max = [np.inf, np.inf, est_sine_k + freq_tol, np.pi]

                popt_sine, _ = curve_fit(self._sine_model, x_roi, y_roi, p0=p0_sine, bounds=(bs_min, bs_max), maxfev=2000)
                sine_k_ref, sine_ph_ref = popt_sine[2], popt_sine[3]
            else:
                sine_k_ref, sine_ph_ref = est_sine_k, 0.0
        except Exception:
            sine_k_ref, sine_ph_ref = est_sine_k, 0.0

        # --- Stage 3: Full Visibility Fit ---
        # Combine results from Stage 1 & 2 as initial guesses
        p0_final = [env_base, env_amp, env_w, env_center, 0.5, sine_k_ref, sine_ph_ref]
        
        # Constrain frequency to Stage 2 result (+/- 10%) to prevent lock jumping
        k_final_tol = 0.1 * sine_k_ref
        
        bounds_min = [-np.inf, 0, 0, 0, 0.0, max(0, sine_k_ref - k_final_tol), -np.pi]
        bounds_max = [np.inf, np.inf, 1.0, len(y), 1.0, sine_k_ref + k_final_tol, np.pi]
        
        try:
            popt, pcov = curve_fit(
                self._full_interference_model, x, y, 
                p0=p0_final, bounds=(bounds_min, bounds_max), maxfev=5000
            )
            
            vis = popt[4]
            sigma = self.calculate_sigma(vis)

            params = {
                'baseline': float(popt[0]),
                'amplitude': float(popt[1]),
                'sinc_width': float(popt[2]),
                'sinc_center': float(popt[3]),
                'visibility': float(popt[4]),
                'sine_k': float(popt[5]),
                'sine_phase': float(popt[6]),
            }

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
                sigma_microns=sigma * 1e6,
                fitted_curve=self._full_interference_model(x, *popt),
                params=params,
                param_errors=param_errors,
                pcov=pcov
            )
            
        except Exception as e:
            return FitResult(success=False, message=f"Fit Failed: {str(e)}")
                

    def calculate_sigma(self, visibility: float) -> float:
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