"""Analysis tables reproducing the paper's quantitative results.

Specification:
1. AUROC per risk model with 95% bootstrap CI resampling whole episodes (500 resamples),
   difference versus score-only, and AURC.
2. Fixed-budget success after k = 1..4 views.
3. Policy table for alpha in {0.05, 0.10, 0.20}: lambda, coverage, failure rate among executed,
   loss, share of episodes with successful executed grasp, mean views.
4. Bound check: 200 random re-splits pooling calibration + test episodes,
   reporting mean test loss, SD, and share of re-splits above alpha for each risk model.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from risk.conformal import (
    evaluate_policy_offline, calibrate_conformal_risk, compute_risk_curve
)
from risk.models import RiskModelSuite


def compute_aurc(risks: np.ndarray, outcomes: np.ndarray, num_lambdas: int = 101) -> float:
    """Compute Area Under the Risk-Coverage Curve (AURC)."""
    _, r_hat, cov_hat, _ = compute_risk_curve(risks, outcomes, num_lambdas=num_lambdas)
    # Sort by coverage ascending
    sort_idx = np.argsort(cov_hat)
    cov_sorted = cov_hat[sort_idx]
    r_sorted = r_hat[sort_idx]
    # Trapezoidal integration
    aurc = float(np.trapz(r_sorted, cov_sorted))
    return aurc


def compute_auroc_table(
    test_features: np.ndarray,
    test_outcomes: np.ndarray,
    risk_suite: RiskModelSuite,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute AUROC with 95% episode-level bootstrap CI and AURC for all risk models."""
    rng = np.random.RandomState(seed)
    N_episodes, T, _ = test_features.shape

    # True failure labels: 1 if failed (outcome == 0), 0 if succeeded
    y_true = (test_outcomes == 0).astype(int)  # (N, T)

    # Predicted risks for each model
    predicted_risks = risk_suite.predict_risk(test_features)  # dict of (N, T)

    models = ["score_only", "dispersion", "cross_view", "all_features"]
    results = []

    # Point estimates
    point_auroc = {}
    point_aurc = {}
    for m in models:
        y_flat = y_true.flatten()
        p_flat = predicted_risks[m].flatten()
        try:
            auc = float(roc_auc_score(y_flat, p_flat))
        except Exception:
            auc = 0.5
        point_auroc[m] = auc
        point_aurc[m] = compute_aurc(predicted_risks[m], test_outcomes)

    # Episode-level bootstrap resamples (resample whole episodes)
    boot_auroc = {m: [] for m in models}
    boot_diff = {m: [] for m in models}

    for b in range(n_bootstrap):
        boot_idx = rng.choice(N_episodes, size=N_episodes, replace=True)
        y_boot = y_true[boot_idx].flatten()
        if len(np.unique(y_boot)) < 2:
            continue

        auc_score_only = 0.5
        for m in models:
            p_boot = predicted_risks[m][boot_idx].flatten()
            try:
                auc_b = float(roc_auc_score(y_boot, p_boot))
            except Exception:
                auc_b = 0.5
            boot_auroc[m].append(auc_b)
            if m == "score_only":
                auc_score_only = auc_b

        for m in models:
            boot_diff[m].append(boot_auroc[m][-1] - auc_score_only)

    for m in models:
        ci_low = float(np.percentile(boot_auroc[m], 2.5))
        ci_high = float(np.percentile(boot_auroc[m], 97.5))

        diff_pt = point_auroc[m] - point_auroc["score_only"]
        diff_low = float(np.percentile(boot_diff[m], 2.5))
        diff_high = float(np.percentile(boot_diff[m], 97.5))

        results.append({
            "Risk Model": m,
            "AUROC": f"{point_auroc[m]:.3f} [{ci_low:.3f}, {ci_high:.3f}]",
            "Diff vs Score-Only": f"{diff_pt:+.3f} [{diff_low:+.3f}, {diff_high:+.3f}]" if m != "score_only" else "-",
            "AURC": f"{point_aurc[m]:.4f}",
        })

    return pd.DataFrame(results)


def compute_fixed_budget_table(test_outcomes: np.ndarray) -> pd.DataFrame:
    """Compute fixed-budget success after k = 1..4 views."""
    N, T = test_outcomes.shape
    rows = []
    for k in range(T):
        # Best sample outcome at view k
        succ_k = test_outcomes[:, k]
        rate = float(np.mean(succ_k))
        std_err = float(np.std(succ_k) / np.sqrt(N))
        rows.append({
            "Budget (Views)": k + 1,
            "Success Rate": f"{rate * 100:.1f}% +/- {std_err * 100:.1f}%",
            "Raw Mean": rate,
            "Std Error": std_err,
        })
    return pd.DataFrame(rows)


def compute_policy_table(
    calib_risks: np.ndarray,
    calib_outcomes: np.ndarray,
    test_risks: np.ndarray,
    test_outcomes: np.ndarray,
    alphas: List[float] = [0.05, 0.10, 0.20],
) -> pd.DataFrame:
    """Compute policy performance metrics on test split for alpha in {0.05, 0.10, 0.20}."""
    rows = []
    for a in alphas:
        # Calibrate lambda_hat on calibration data
        lambda_hat, _, _ = calibrate_conformal_risk(calib_risks, calib_outcomes, alpha=a)

        # Evaluate calibrated policy on test data
        res = evaluate_policy_offline(test_risks, test_outcomes, lambda_hat)

        rows.append({
            "Alpha": a,
            "Lambda*": f"{lambda_hat:.3f}",
            "Coverage": f"{res['coverage'] * 100:.1f}%",
            "Failure Rate (Exec)": f"{res['failure_rate_executed'] * 100:.1f}%",
            "Loss": f"{res['mean_loss']:.3f}",
            "Success (Episodes)": f"{res['success_rate_all'] * 100:.1f}%",
            "Mean Views": f"{res['mean_views']:.2f}",
        })
    return pd.DataFrame(rows)


def compute_bound_check(
    all_risks: Dict[str, np.ndarray],
    all_outcomes: np.ndarray,
    alpha: float = 0.10,
    n_resplits: int = 200,
    calib_size: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Pool calibration + test episodes and run 200 random re-splits for EACH risk model."""
    rng = np.random.RandomState(seed)
    N_total = len(all_outcomes)
    calib_n = min(calib_size, int(N_total * 0.5))

    models = ["score_only", "dispersion", "cross_view", "all_features"]
    results = []

    for m in models:
        m_risks = all_risks[m]
        losses_split = []

        for _ in range(n_resplits):
            perm = rng.permutation(N_total)
            c_idx = perm[:calib_n]
            t_idx = perm[calib_n:]

            c_r, c_y = m_risks[c_idx], all_outcomes[c_idx]
            t_r, t_y = m_risks[t_idx], all_outcomes[t_idx]

            lambda_hat, _, _ = calibrate_conformal_risk(c_r, c_y, alpha=alpha)
            res = evaluate_policy_offline(t_r, t_y, lambda_hat)
            losses_split.append(res["mean_loss"])

        losses_arr = np.array(losses_split)
        mean_l = float(np.mean(losses_arr))
        sd_l = float(np.std(losses_arr))
        viol_share = float(np.mean(losses_arr > alpha))

        results.append({
            "Risk Model": m,
            "Alpha": alpha,
            "Mean Test Loss": f"{mean_l:.4f}",
            "SD": f"{sd_l:.4f}",
            "Share Above Alpha": f"{viol_share * 100:.1f}%",
        })

    return pd.DataFrame(results)
