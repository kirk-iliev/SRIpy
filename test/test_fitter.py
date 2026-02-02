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
