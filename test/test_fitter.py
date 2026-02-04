import os
import sys
import numpy as np
# Ensure project root is on sys.path so imports like `analysis.fitter` work when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.fitter import InterferenceFitter


def test_fit_returns_all_parameters_and_errors():
    np.random.seed(0)
    fitter = InterferenceFitter()

    x = np.arange(0, 1024)
    true_params = {
        'baseline': 10.0,
        'amplitude': 100.0,
        'sinc_width': 0.02,
        'sinc_center': 512.0,
        'visibility': 0.6,
        'sine_k': 0.05,
        'sine_phase': 0.1
    }

    y = fitter._full_interference_model(
        x,
        true_params['baseline'],
        true_params['amplitude'],
        true_params['sinc_width'],
        true_params['sinc_center'],
        true_params['visibility'],
        true_params['sine_k'],
        true_params['sine_phase']
    )

    # Add small noise but keep SNR high
    y += np.random.normal(scale=0.5, size=y.shape)

    res = fitter.fit(y)

    assert res.success is True
    assert res.params is not None
    # Ensure all keys are present and finite
    for k in true_params.keys():
        assert k in res.params
        assert np.isfinite(res.params[k])

    # Visibility should be reasonably close (within 20%)
    assert abs(res.params['visibility'] - true_params['visibility']) < 0.2

    # Param errors should be present and finite
    assert res.param_errors is not None
    for v in res.param_errors.values():
        assert np.isfinite(v)

    # Check the fitted curve matches the noise-free signal well (RMSE small)
    y_true = fitter._full_interference_model(
        x,
        true_params['baseline'],
        true_params['amplitude'],
        true_params['sinc_width'],
        true_params['sinc_center'],
        true_params['visibility'],
        true_params['sine_k'],
        true_params['sine_phase']
    )
    rmse = np.sqrt(np.mean((res.fitted_curve - y_true) ** 2))
    # RMSE should be much less than signal amplitude
    assert rmse < 5.0


def test_four_stage_envelope_fits_correctly():
    """Stage 0→1: Gaussian → Sinc² envelope."""
    np.random.seed(42)
    fitter = InterferenceFitter()
    
    x = np.arange(1024)
    # Pure envelope: Sinc² with no fringes (visibility=0)
    y = fitter._sinc_sq_envelope(x, baseline=20.0, amp=150.0, width=0.015, center=512.0)
    y += np.random.normal(scale=1.0, size=len(y))
    
    res = fitter.fit(y)
    
    assert res.success is True
    # Envelope fit should recover center and amplitude well
    assert abs(res.params['sinc_center'] - 512.0) < 20  # Center within 20 pixels
    assert abs(res.params['amplitude'] - 150.0) < 30  # Amplitude within 30 units


def test_four_stage_fft_frequency_locking():
    """Stage 2→3: FFT frequency estimate is locked with ±10% constraint."""
    np.random.seed(123)
    fitter = InterferenceFitter()
    
    x = np.arange(2048)
    # Target frequency: 0.08 (should correspond to ~125 period)
    true_k = 0.08
    y = fitter._full_interference_model(
        x,
        baseline=15.0,
        amp=120.0,
        sinc_w=0.02,
        sinc_x0=1024.0,
        visibility=0.7,
        sine_freq=true_k,
        sine_phase=0.5
    )
    y += np.random.normal(scale=2.0, size=len(y))
    
    res = fitter.fit(y)
    
    assert res.success is True
    # Frequency should be recovered within FFT resolution + Stage 2 constraints
    assert abs(res.params['sine_k'] - true_k) < 0.02  # Within ±0.02


def test_high_noise_robustness():
    """Verify 4-stage process doesn't diverge on noisy data."""
    np.random.seed(99)
    fitter = InterferenceFitter()
    
    x = np.arange(1024)
    y = fitter._full_interference_model(
        x, 30.0, 100.0, 0.025, 512.0, 0.5, 0.06, 0.2
    )
    # High noise: SNR ~1:1
    y += np.random.normal(scale=80.0, size=len(y))
    
    res = fitter.fit(y)
    
    # Should still succeed with reasonable results
    assert res.success is True
    assert 0 <= res.params['visibility'] <= 1.0
    assert res.sigma_microns > 0
    # Shouldn't have blown up to infinity
    for v in res.params.values():
        assert np.isfinite(v)


def test_edge_case_low_signal():
    """Low signal should gracefully fail."""
    fitter = InterferenceFitter()
    y = np.ones(1024) * 10.0  # Flat signal, diff < 50
    res = fitter.fit(y)
    assert res.success is False
    assert "Low Signal" in res.message


def test_edge_case_all_nan():
    """NaN data should be sanitized and fail gracefully."""
    fitter = InterferenceFitter()
    y = np.full(1024, np.nan)
    res = fitter.fit(y)
    assert res.success is False


def test_visibility_physically_valid():
    """Recovered visibility should always be 0 ≤ V ≤ 1."""
    np.random.seed(7)
    fitter = InterferenceFitter()
    
    for vis_true in [0.0, 0.3, 0.6, 0.95]:
        x = np.arange(1024)
        y = fitter._full_interference_model(
            x, 20.0, 100.0, 0.018, 512.0, vis_true, 0.07, 0.0
        )
        y += np.random.normal(scale=1.5, size=len(y))
        res = fitter.fit(y)
        
        if res.success:
            assert 0.0 <= res.params['visibility'] <= 1.0
