import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class FitResult:
    """Standardized container for interference fit results."""
    success: bool
    visibility: float = 0.0
    raw_visibility: float = 0.0
    sigma_microns: float = 0.0
    fitted_curve: Optional[np.ndarray] = None
    params: Optional[Dict[str, float]] = None
    param_errors: Optional[Dict[str, float]] = None
    pcov: Optional[np.ndarray] = None
    fit_x: Optional[np.ndarray] = None
    message: str = ""
    peak_idx: Optional[int] = None
    valley_idx: Optional[int] = None

class InterferenceFitter:
    def __init__(self, wavelength: float = 550e-9, slit_separation: float = 0.05, distance: float = 16.5, min_signal: float = 50.0):
        import logging
        self.logger = logging.getLogger(__name__)
        self.wavelength = wavelength
        self.slit_sep = slit_separation
        self.distance = distance
        self.min_signal = min_signal

    # --- Data Preparation ---

    # --- Model Definitions ---
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


    # --- Contrast Calculation ---
    def _calculate_raw_contrast(self, y: np.ndarray) -> tuple:
        """
        Calculates simple (Imax - Imin) / (Imax + Imin) visibility.
        Finds the global peak and the lowest valley in its immediate neighborhood.
        Returns: (visibility, peak_idx, valley_idx)
        """
        try:
            # 1. Find the Peak (Imax)
            peak_idx = np.argmax(y)
            i_max = y[peak_idx]
            signal_range = np.max(y) - np.min(y)
            valleys, _ = find_peaks(-y, prominence=signal_range * 0.05)
            if len(valleys) == 0:
                return 0.0, peak_idx, None

            left_candidates = valleys[valleys < peak_idx]
            right_candidates = valleys[valleys > peak_idx]

            i_mins = []
            valley_indices = []
            if len(left_candidates) > 0:
                left_valley_idx = left_candidates[-1]
                i_mins.append(y[left_valley_idx])
                valley_indices.append(left_valley_idx)

            if len(right_candidates) > 0:
                right_valley_idx = right_candidates[0]
                i_mins.append(y[right_valley_idx])
                valley_indices.append(right_valley_idx)

            if not i_mins:
                i_min = np.min(y)
                valley_idx = np.argmin(y)
            else:
                i_min = np.mean(i_mins)
                # Use the valley closest to the peak
                valley_idx = valley_indices[np.argmin(np.abs(np.array(valley_indices) - peak_idx))]

            denominator = i_max + i_min
            if denominator == 0:
                return 0.0, peak_idx, valley_idx

            return (i_max - i_min) / denominator, peak_idx, valley_idx

        except Exception:
            return 0.0, None, None

    # --- Fitting Logic ---
    def fit(self, lineout: np.ndarray, roi_hint: Optional[Tuple[int, int]] = None) -> FitResult:
        """
        Fit interference pattern using multi-stage approach.

        Follows the MATLAB intdisplay_sinc2 strategy:
          0. Gaussian fit on FULL data → robust center finding
          1. Sinc² envelope fit on FULL data → baseline + envelope params
          2. FFT on FULL data → fringe frequency estimate
          3. Sine fit on narrow region around center → phase/freq refinement
          4. Full model fit on centered region → visibility extraction

        This ensures the fit always tracks the beam center regardless of the
        display ROI position, eliminating jitter and ROI-sensitivity.

        Args:
            lineout: 1D interference pattern (ideally full extent).
            roi_hint: Optional (start, stop) pixel range from display ROI.
                      Determines fit region WIDTH only; the region is always
                      centered on the internally-found peak, not the ROI center.
                      If None, fits over the full input range (backward compat).
        """
        # Sanitize Input
        y_full = np.nan_to_num(np.asarray(lineout, dtype=float))
        x_full = np.arange(len(y_full))
        n_full = len(y_full)

        # --- Smoothed peak detection (reduces noise-induced center jumps) ---
        smooth_size = max(3, min(15, n_full // 100))
        y_smooth = uniform_filter1d(y_full, size=smooth_size)
        peak_idx_rough = int(np.argmax(y_smooth))
        peak_val = float(y_full[peak_idx_rough])
        min_val = float(np.min(y_full))
        signal_range = peak_val - min_val

        if signal_range < self.min_signal:
            return FitResult(success=False, message="Low Signal", peak_idx=peak_idx_rough)

        # === Stage 0: Gaussian on Full Data → Robust Center Finding ===
        half_max = (peak_val + min_val) / 2
        above_half = np.where(y_full > half_max)[0]
        est_width = float((above_half[-1] - above_half[0]) / 2.0) if len(above_half) > 1 else 50.0
        est_width = max(est_width, 5.0)

        p0_gauss = [min_val, signal_range, peak_idx_rough, est_width]
        try:
            popt_g, _ = curve_fit(
                self._gaussian, x_full, y_full, p0=p0_gauss,
                bounds=([-np.inf, 0, 0, 0.1], [np.inf, np.inf, n_full, n_full]),
                maxfev=1000
            )
            center_guess = float(popt_g[2])
            width_guess = float(popt_g[3])
        except Exception as e:
            self.logger.debug(f"Gaussian fit fallback: {e}")
            center_guess = float(peak_idx_rough)
            width_guess = est_width

        # === Stage 1: Sinc² Envelope on Full Data → Lock Baseline ===
        # (Like MATLAB: fits sinc² to entire IMG.IntPattern before cropping)
        est_sinc_w = 2.0 / max(width_guess * 2, 1.0)
        p0_env = [min_val, signal_range, est_sinc_w, center_guess]

        try:
            popt_env, _ = curve_fit(
                self._sinc_sq_envelope, x_full, y_full, p0=p0_env, maxfev=2000
            )
            env_base, env_amp, env_w, env_center = [float(v) for v in popt_env]
        except Exception as e:
            self.logger.debug(f"Envelope fit fallback: {e}")
            env_base, env_amp, env_w, env_center = min_val, signal_range, est_sinc_w, center_guess

        # Update center from envelope fit (more robust than Gaussian alone)
        center = env_center

        # === Determine Fit Region (centered on found peak, not ROI center) ===
        if roi_hint is not None:
            # Use ROI width, but CENTER on found peak
            fit_half_width = int((roi_hint[1] - roi_hint[0]) / 2)
            fit_half_width = max(fit_half_width, 50)
        else:
            # No ROI hint: fit full range (backward compat with tests)
            fit_half_width = n_full

        center_idx = int(round(center))
        center_idx = max(0, min(n_full - 1, center_idx))
        fit_start = max(0, center_idx - fit_half_width)
        fit_stop = min(n_full, center_idx + fit_half_width)

        if fit_stop - fit_start < 30:
            fit_start = max(0, center_idx - 50)
            fit_stop = min(n_full, center_idx + 50)

        y = y_full[fit_start:fit_stop]
        x = np.arange(fit_start, fit_stop)

        # Raw visibility on centered region
        raw_vis, peak_idx_local, valley_idx_local = self._calculate_raw_contrast(y)
        peak_idx_abs = (peak_idx_local + fit_start) if peak_idx_local is not None else None
        valley_idx_abs = (valley_idx_local + fit_start) if valley_idx_local is not None else None

        # === Stage 2: FFT on Full Lineout → Frequency Estimate ===
        # (Like MATLAB: ffts=abs(fft(s)) on full waveform, not the crop)
        try:
            y_ac = y_full - np.mean(y_full)
            window = np.hanning(n_full)
            yf = np.fft.rfft(y_ac * window)
            xf = np.fft.rfftfreq(n_full)
            fft_mag = np.abs(yf)

            # Mask envelope frequencies
            cutoff_k = 1.5 * abs(env_w) if abs(env_w) > 1e-6 else 0.01
            cutoff_idx = int((cutoff_k * n_full) / (2 * np.pi))
            cutoff_idx = max(5, cutoff_idx)
            fft_mag[0:cutoff_idx] = 0

            dominant_idx = int(np.argmax(fft_mag))
            est_sine_k = float(2 * np.pi * xf[dominant_idx]) if dominant_idx > 0 else 0.3
        except Exception as e:
            self.logger.debug(f"FFT frequency estimate fallback: {e}")
            est_sine_k = 0.3

        if est_sine_k < 0.01:
            est_sine_k = 0.3

        # === Stage 3: Sine Fit on Narrow Region Around Center ===
        # (Like MATLAB: npts=50 for sinusoid fitting around the found peak)
        npts_sine = 50
        sine_start = max(fit_start, center_idx - npts_sine)
        sine_stop = min(fit_stop, center_idx + npts_sine)

        sine_k_ref = est_sine_k
        sine_ph_ref = 0.0

        try:
            if sine_stop - sine_start > 10:
                y_sine = y_full[sine_start:sine_stop]
                x_sine = np.arange(sine_start, sine_stop)

                p0_sine = [float(np.mean(y_sine)),
                           float((np.max(y_sine) - np.min(y_sine)) / 2),
                           est_sine_k, 0.0]
                freq_tol = 0.2 * max(est_sine_k, 0.01)
                bs_min = [-np.inf, 0, max(0.001, est_sine_k - freq_tol), -np.pi]
                bs_max = [np.inf, np.inf, est_sine_k + freq_tol, np.pi]

                popt_sine, _ = curve_fit(
                    self._sine_model, x_sine, y_sine, p0=p0_sine,
                    bounds=(bs_min, bs_max), maxfev=2000
                )
                sine_k_ref = float(popt_sine[2])
                sine_ph_ref = float(popt_sine[3])
        except Exception as e:
            self.logger.debug(f"Sine fit refinement fallback: {e}")

        # === Stage 4: Full Visibility Fit on Centered Region ===
        # MATLAB approach: lock baseline near envelope fit value to prevent
        # baseline/visibility parameter trading that causes instability
        p0_final = [env_base, env_amp, env_w, center, 0.5, sine_k_ref, sine_ph_ref]

        k_final_tol = 0.1 * max(sine_k_ref, 0.01)
        base_tol = max(abs(env_base) * 0.5, 20.0)
        center_tol = 30.0

        bounds_min = [
            env_base - base_tol,                         # baseline (locked near envelope)
            0,                                           # amplitude
            0,                                           # sinc_width
            max(float(fit_start), center - center_tol),  # center (tight)
            0.0,                                         # visibility
            max(0.001, sine_k_ref - k_final_tol),        # sine_k
            -np.pi                                       # sine_phase
        ]
        bounds_max = [
            env_base + base_tol,
            np.inf,
            5.0,
            min(float(fit_stop), center + center_tol),
            1.0,
            sine_k_ref + k_final_tol,
            np.pi
        ]

        try:
            popt, pcov = curve_fit(
                self._full_interference_model, x, y,
                p0=p0_final, bounds=(bounds_min, bounds_max), maxfev=5000
            )

            if not all(np.isfinite(popt)):
                return FitResult(success=False, message="Fit converged but produced non-finite parameters",
                                 peak_idx=peak_idx_abs, valley_idx=valley_idx_abs)

            vis = float(np.clip(popt[4], 0.0, 1.0))
            sigma = self.calculate_sigma(vis)

            params = {
                'baseline': float(popt[0]),
                'amplitude': float(popt[1]),
                'sinc_width': float(popt[2]),
                'sinc_center': float(popt[3]),
                'visibility': vis,
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
            except Exception as e:
                self.logger.debug(f"Param error estimation failed: {e}")

            try:
                fitted_curve = self._full_interference_model(x, *popt)
                if not all(np.isfinite(fitted_curve)):
                    self.logger.warning("Fitted curve contains non-finite values")
                    fitted_curve = np.clip(fitted_curve, -1e10, 1e10)
            except Exception as e:
                self.logger.warning(f"Error computing fitted curve: {e}")
                fitted_curve = np.full_like(y, np.nan)

            # === Evaluate fit over entire lineout range for display ===
            # This ensures the fit curve shows the interference structure across
            # the entire field of view, not just the cropped fitting region
            try:
                fitted_curve_full = self._full_interference_model(x_full, *popt)
                if not all(np.isfinite(fitted_curve_full)):
                    self.logger.warning("Full fitted curve contains non-finite values")
                    fitted_curve_full = np.clip(fitted_curve_full, -1e10, 1e10)
            except Exception as e:
                self.logger.warning(f"Error computing full fitted curve: {e}")
                fitted_curve_full = np.full_like(y_full, np.nan)

            return FitResult(
                success=True,
                visibility=vis,
                raw_visibility=raw_vis,
                sigma_microns=sigma * 1e6,
                fitted_curve=fitted_curve_full,  # Use full range fit for display
                fit_x=x_full.copy(),  # Use full x range for display
                params=params,
                param_errors=param_errors,
                pcov=pcov,
                peak_idx=peak_idx_abs,
                valley_idx=valley_idx_abs
            )

        except RuntimeError as e:
            return FitResult(success=False, message=f"Fit Failed (convergence): {str(e)[:100]}",
                             peak_idx=peak_idx_abs, valley_idx=valley_idx_abs)
        except ValueError as e:
            return FitResult(success=False, message=f"Fit Failed (invalid data): {str(e)[:100]}",
                             peak_idx=peak_idx_abs, valley_idx=valley_idx_abs)
        except Exception as e:
            self.logger.error(f"Unexpected fit error: {e}", exc_info=True)
            return FitResult(success=False, message=f"Fit Failed (unexpected): {str(type(e).__name__)[:50]}",
                             peak_idx=peak_idx_abs, valley_idx=valley_idx_abs)


    def calculate_sigma(self, visibility: float) -> float:
        """Calculate beam sigma from visibility with robust error handling."""
        # Validate input
        if not np.isfinite(visibility):
            return 0.0

        # Clamp visibility to valid range (avoid math domain errors)
        visibility = np.clip(visibility, 0.001, 0.999)

        # Return 0 only if visibility extremely low (no useful signal)
        if visibility <= 0.001:
            return 0.0

        coeff = (self.wavelength * self.distance) / (np.pi * self.slit_sep)

        try:
            # Safely compute sigma
            arg = 0.5 * np.log(1.0 / visibility)
            if arg <= 0:
                return 0.0
            sigma = coeff * np.sqrt(arg)

            # Validate result
            if not np.isfinite(sigma) or sigma < 0:
                return 0.0
            return sigma

        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            self.logger.debug(f"calculate_sigma: math error with visibility={visibility}: {e}")
            return 0.0