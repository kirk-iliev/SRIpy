import numpy as np
from scipy.optimize import curve_fit

class InterferenceFitter:
    def __init__(self, wavelength=550e-9, slit_separation=0.05, distance=16.5):
        self.wavelength = wavelength
        self.slit_sep = slit_separation
        self.distance = distance  # 'L' in your matlab script

    def get_lineout(self, image):
        """
        Replicates MATLAB: s=sum(f,1);
        Sum vertically to get a 1D horizontal profile.
        """
        # Ensure it's a 2D float array
        if image.ndim == 3:
            image = image.mean(axis=2)
        return np.sum(image, axis=0)

    def model_function(self, x, baseline, sinc_amp, sinc_width, sinc_offset, visibility, sine_freq, sine_phase):
        """
        The Physics Equation from your MATLAB 'visfit' function.
        I(x) = Baseline + [ Sinc^2_Envelope ] * [ 1 + V * sin(kx + phi) ]
        """
        # Avoid divide by zero
        x_centered = x - sinc_offset
        # sinc(x) in numpy is sin(pi*x)/(pi*x), so we scale strictly to match matlab's sin(ax)/ax
        # MATLAB: sin(w*x)/(w*x)
        # We handle the singularity at x=0 explicitly or use np.sinc which handles it but has a pi factor.
        # Let's use raw sin/x to match MATLAB exactly.
        
        # Add epsilon to avoid nan at 0
        x_safe = x_centered + 1e-9 
        
        sinc_term = np.sin(sinc_width * x_safe) / (sinc_width * x_safe)
        sinc_sq = sinc_amp * (sinc_term ** 2)
        
        sine_term = 1 + visibility * np.sin(sine_freq * x + sine_phase)
        
        return baseline + sinc_sq * sine_term

    def fit(self, lineout):
        """
        Performs the non-linear least squares fit.
        """
        x = np.arange(len(lineout))
        
        # --- 1. Guess Initial Parameters (Heuristics) ---
        peak_y = np.max(lineout)
        peak_x = np.argmax(lineout)
        baseline = np.min(lineout)
        
        # Guess from MATLAB defaults:
        p0 = [
            baseline,           # Baseline
            peak_y - baseline,  # Sinc Amp
            0.02,               # Sinc Width (B)
            peak_x,             # Sinc Offset (x0)
            0.5,                # Visibility (V)
            0.3,                # Sine Freq (guess, usually found via FFT)
            0.0                 # Sine Phase
        ]
        
        # --- 2. Refine Frequency Guess using FFT (Optional but robust) ---
        # (We can skip this for now or add it if the fit struggles)

        try:
            # Perform Fit
            # bounds: (min_vals, max_vals) to keep visibility between 0 and 1, etc.
            params, covariance = curve_fit(
                self.model_function, 
                x, 
                lineout, 
                p0=p0,
                maxfev=5000
            )
            
            # Extract Results
            results = {
                'baseline': params[0],
                'amplitude': params[1],
                'visibility': abs(params[4]), # V must be positive
                'sigma': self.calculate_sigma(abs(params[4])),
                'fitted_curve': self.model_function(x, *params),
                'params': params
            }
            return results

        except RuntimeError:
            return None # Fit failed

    def calculate_sigma(self, visibility):
        """
        Replicates MATLAB 'fit_sigma' function.
        sigma = coeff * sqrt(0.5 * log(1/V))
        """
        if visibility <= 0 or visibility >= 1:
            return 0.0
            
        coeff = (self.wavelength * self.distance) / (np.pi * self.slit_sep)
        sigma = coeff * np.sqrt(0.5 * np.log(1.0 / visibility))
        return sigma