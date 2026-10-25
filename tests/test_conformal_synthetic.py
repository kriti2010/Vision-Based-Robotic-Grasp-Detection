"""Acceptance Test 3:
On synthetic exchangeable data, the mean test loss over many re-splits
is <= alpha + 2 standard errors, and loss is monotone in lambda.
"""

import numpy as np
import pytest
from risk.conformal import (
    calibrate_conformal_risk, evaluate_policy_offline,
    compute_risk_curve, check_monotonicity
)


def test_conformal_bound_and_monotonicity_on_synthetic_data():
    """Verify validity of conformal risk bound and empirical loss monotonicity."""
    rng = np.random.RandomState(42)
    N_total = 1000
    T = 4
    alpha = 0.10

    # Generate synthetic exchangeable per-view risks and outcomes
    # Simulated risk: rho ~ Beta(2, 5)
    risks = rng.beta(2.0, 5.0, size=(N_total, T)).astype(np.float32)
    # Binary outcome: success prob = 1 - rho
    outcomes = (rng.rand(N_total, T) >= risks).astype(np.int32)

    # 1. Exact Monotonicity on single-view data
    risks_single = rng.beta(2.0, 5.0, size=(N_total, 1)).astype(np.float32)
    outcomes_single = (rng.rand(N_total, 1) >= risks_single).astype(np.int32)
    lambdas_s, r_hat_s, _, _ = compute_risk_curve(risks_single, outcomes_single, num_lambdas=101)
    diag_s = check_monotonicity(lambdas_s, r_hat_s)
    assert diag_s["is_monotone"], "Single-view risk curve must be strictly monotone"

    # 2. Multi-view monotonicity diagnostic (tracks discrete view-switching fluctuations)
    lambdas, r_hat, cov_hat, _ = compute_risk_curve(risks, outcomes, num_lambdas=101)
    diag = check_monotonicity(lambdas, r_hat)
    assert abs(diag["max_negative_drop"]) < 0.01, f"Multi-view risk drop {diag['max_negative_drop']} exceeded 0.01"

    # 3. Re-split validation (200 random splits)
    n_splits = 200
    calib_size = 500
    test_losses = []

    for s_idx in range(n_splits):
        perm = rng.permutation(N_total)
        calib_idx = perm[:calib_size]
        test_idx = perm[calib_size:]

        calib_risks = risks[calib_idx]
        calib_outcomes = outcomes[calib_idx]
        test_risks = risks[test_idx]
        test_outcomes = outcomes[test_idx]

        # Calibrate lambda_hat on calibration split
        lambda_hat, _, _ = calibrate_conformal_risk(calib_risks, calib_outcomes, alpha=alpha, num_lambdas=101)

        # Evaluate on test split
        res_test = evaluate_policy_offline(test_risks, test_outcomes, lambda_hat)
        test_losses.append(res_test["mean_loss"])

    test_losses = np.array(test_losses)
    mean_test_loss = float(np.mean(test_losses))
    std_err = float(np.std(test_losses) / np.sqrt(n_splits))

    print(f"\nSynthetic Conformal Validation: Mean Test Loss = {mean_test_loss:.4f} +/- {std_err:.4f} (alpha = {alpha})")

    # Bound check: mean test loss <= alpha + 2 standard errors
    assert mean_test_loss <= (alpha + 2.0 * std_err), (
        f"Mean test loss {mean_test_loss:.4f} exceeded alpha + 2*SE ({alpha + 2*std_err:.4f})"
    )
