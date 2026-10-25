"""Conformal risk control, Learn-then-Test, and policy evaluation.

Specification:
- Policy: execute at first view with rho <= lambda, else next view, abstain after T.
  Loss = 1 if executed and failed, else 0 (abstain = 0).
- Offline loss computation from recorded per-view risks and outcomes.
- Conformal risk control: grid of 201 lambdas in [0, 1].
  lambda_hat = sup{lambda : n/(n+1)*Rhat(lambda) + 1/(n+1) <= alpha}.
- Learn-then-Test (fixed-sequence binomial tail, delta = 0.1).
- Monotonicity diagnostic.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.stats import binom


def evaluate_policy_offline(
    risks: np.ndarray,
    outcomes: np.ndarray,
    lambda_val: float,
) -> Dict[str, Any]:
    """Evaluate execute/re-view/abstain policy offline for a fixed lambda.
    
    Args:
        risks: (N, T) predicted failure risk rho in [0, 1]
        outcomes: (N, T) binary simulator outcomes (1=success, 0=failure)
        lambda_val: execution threshold
    Returns:
        dict with:
        - losses: (N,) binary loss
        - executed: (N,) bool
        - view_counts: (N,) int
        - successes: (N,) bool
        - mean_loss: float
        - coverage: float
        - failure_rate_executed: float
        - success_rate_all: float
        - mean_views: float
    """
    N, T = risks.shape
    losses = np.zeros(N, dtype=np.float32)
    executed = np.zeros(N, dtype=bool)
    successes = np.zeros(N, dtype=bool)
    view_counts = np.full(N, T, dtype=np.int32)

    for i in range(N):
        # Find first view with rho <= lambda_val
        exec_view = -1
        for t in range(T):
            if risks[i, t] <= lambda_val:
                exec_view = t
                break

        if exec_view >= 0:
            executed[i] = True
            view_counts[i] = exec_view + 1
            if outcomes[i, exec_view] == 1:
                successes[i] = True
                losses[i] = 0.0
            else:
                losses[i] = 1.0  # Executed and failed
        else:
            # Abstained
            executed[i] = False
            view_counts[i] = T
            losses[i] = 0.0  # Abstain = 0

    cov = float(np.mean(executed))
    loss_mean = float(np.mean(losses))
    n_exec = int(np.sum(executed))
    fail_rate = float(np.sum(losses) / max(1, n_exec)) if n_exec > 0 else 0.0
    succ_rate = float(np.mean(successes))
    views_mean = float(np.mean(view_counts))

    return {
        "losses": losses,
        "executed": executed,
        "view_counts": view_counts,
        "successes": successes,
        "mean_loss": loss_mean,
        "coverage": cov,
        "failure_rate_executed": fail_rate,
        "success_rate_all": succ_rate,
        "mean_views": views_mean,
    }


def compute_risk_curve(
    risks: np.ndarray,
    outcomes: np.ndarray,
    num_lambdas: int = 201,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute risk, coverage, and losses across grid of 201 lambdas in [0, 1]."""
    lambdas = np.linspace(0.0, 1.0, num_lambdas)
    N = len(risks)
    r_hat = np.zeros(num_lambdas, dtype=np.float64)
    cov_hat = np.zeros(num_lambdas, dtype=np.float64)
    loss_matrix = np.zeros((num_lambdas, N), dtype=np.float32)

    for idx, l in enumerate(lambdas):
        res = evaluate_policy_offline(risks, outcomes, l)
        r_hat[idx] = res["mean_loss"]
        cov_hat[idx] = res["coverage"]
        loss_matrix[idx] = res["losses"]

    return lambdas, r_hat, cov_hat, loss_matrix


def check_monotonicity(lambdas: np.ndarray, r_hat: np.ndarray) -> Dict[str, Any]:
    """Diagnostic for monotonicity of empirical risk with respect to lambda."""
    diffs = np.diff(r_hat)
    violations = np.sum(diffs < -1e-6)
    max_drop = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    is_monotone = bool(violations == 0)
    return {
        "is_monotone": is_monotone,
        "num_violations": int(violations),
        "max_negative_drop": max_drop,
    }


def calibrate_conformal_risk(
    calib_risks: np.ndarray,
    calib_outcomes: np.ndarray,
    alpha: float,
    num_lambdas: int = 201,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Conformal Risk Control calibration.
    
    lambda_hat = sup{lambda : n / (n + 1) * Rhat(lambda) + 1 / (n + 1) <= alpha}
    """
    n = len(calib_risks)
    lambdas, r_hat, cov_hat, _ = compute_risk_curve(calib_risks, calib_outcomes, num_lambdas=num_lambdas)

    # Conformal adjusted bound: n / (n + 1) * R_hat + 1 / (n + 1) <= alpha
    adjusted_r_hat = (n / (n + 1.0)) * r_hat + (1.0 / (n + 1.0))
    valid_mask = adjusted_r_hat <= alpha

    if not np.any(valid_mask):
        # Abstain policy
        lambda_hat = 0.0
    else:
        lambda_hat = float(lambdas[valid_mask][-1])

    return lambda_hat, lambdas, r_hat


def calibrate_learn_then_test(
    calib_risks: np.ndarray,
    calib_outcomes: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    num_lambdas: int = 201,
) -> float:
    """Learn-then-Test (LTT) risk control using fixed-sequence Binomial testing.
    
    Null hypothesis: R(lambda) > alpha.
    p-value from binomial tail: binom.cdf(n * Rhat, n, alpha).
    """
    n = len(calib_risks)
    lambdas, r_hat, _, _ = compute_risk_curve(calib_risks, calib_outcomes, num_lambdas=num_lambdas)

    valid_lambdas = []
    for idx, l in enumerate(lambdas):
        k = int(np.round(n * r_hat[idx]))
        pval = float(binom.cdf(k, n, alpha))
        if pval <= delta:
            valid_lambdas.append(l)
        else:
            # Fixed sequence testing stops at first rejection failure
            break

    if len(valid_lambdas) == 0:
        return 0.0
    return float(valid_lambdas[-1])
