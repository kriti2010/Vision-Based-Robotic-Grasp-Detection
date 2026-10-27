"""Risk calibration and threshold selection for active stopping rule."""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class CalibrationRecord:
    """Validation record containing uncertainty scores and ground-truth execution success."""
    U_t: float
    rho_t: float
    success: bool
    top_score: float


def compute_risk_coverage(
    records: List[CalibrationRecord],
    tau_U: float,
    tau_rho: float
) -> Tuple[float, float, int]:
    """Compute empirical coverage and risk (error rate) given thresholds.
    
    Returns:
        (coverage, risk, num_executed)
    """
    total = len(records)
    if total == 0:
        return 0.0, 0.0, 0

    # Decision rule: execute if U_t <= tau_U and rho_t >= tau_rho
    executed = [r for r in records if (r.U_t <= tau_U and r.rho_t >= tau_rho)]
    num_exec = len(executed)
    
    if num_exec == 0:
        return 0.0, 0.0, 0

    coverage = num_exec / total
    successful_exec = sum(1 for r in executed if r.success)
    empirical_success_rate = successful_exec / num_exec
    risk = 1.0 - empirical_success_rate

    return coverage, risk, num_exec


def calibrate_stopping_thresholds(
    records: List[CalibrationRecord],
    target_success: float = 0.85,
    u_grid_size: int = 25,
    rho_grid_size: int = 25
) -> Dict[str, Any]:
    """Search for the loosest (tau_U, tau_rho) that achieves target_success (risk <= 1 - target_success).
    
    'Loosest' means maximizing empirical coverage on the validation set.
    """
    if len(records) == 0:
        return {
            "tau_U": 0.05,
            "tau_rho": 0.30,
            "coverage": 0.0,
            "risk": 0.0,
            "success_rate": 0.0
        }

    all_u = [r.U_t for r in records]
    all_rho = [r.rho_t for r in records]

    # Candidate grids
    u_candidates = np.linspace(min(all_u), max(all_u) + 0.01, u_grid_size)
    rho_candidates = np.linspace(max(0.05, min(all_rho)), min(1.0, max(all_rho)), rho_grid_size)

    max_target_risk = 1.0 - target_success

    best_tau_U = float(np.median(all_u))
    best_tau_rho = float(np.median(all_rho))
    best_coverage = -1.0
    best_risk = 1.0

    # Sweep grid
    for u_val in u_candidates:
        for rho_val in rho_candidates:
            cov, risk, n_exec = compute_risk_coverage(records, u_val, rho_val)
            if n_exec >= 2 and risk <= max_target_risk:
                if cov > best_coverage:
                    best_coverage = cov
                    best_risk = risk
                    best_tau_U = float(u_val)
                    best_tau_rho = float(rho_val)

    # Fallback if target was too stringent for validation sample
    if best_coverage < 0.0:
        # Pick threshold that minimizes risk
        min_risk = 1.01
        for u_val in u_candidates:
            for rho_val in rho_candidates:
                cov, risk, n_exec = compute_risk_coverage(records, u_val, rho_val)
                if n_exec >= 1 and risk < min_risk:
                    min_risk = risk
                    best_coverage = cov
                    best_risk = risk
                    best_tau_U = float(u_val)
                    best_tau_rho = float(rho_val)

        if best_coverage < 0.0:
            best_tau_U = float(np.median(all_u))
            best_tau_rho = float(np.median(all_rho))
            cov, risk, _ = compute_risk_coverage(records, best_tau_U, best_tau_rho)
            best_coverage = max(0.0, cov)
            best_risk = risk

    return {
        "tau_U": best_tau_U,
        "tau_rho": best_tau_rho,
        "coverage": best_coverage,
        "risk": best_risk,
        "success_rate": 1.0 - best_risk
    }


def compute_risk_coverage_curve(
    records: List[CalibrationRecord],
    num_points: int = 50
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute risk-coverage trade-off curve and Area Under Risk-Coverage curve (RC-AUC).
    
    Returns:
        (coverages, risks, rc_auc)
    """
    if len(records) == 0:
        return np.array([0.0]), np.array([0.0]), 0.0

    # Sort records by ambiguity score U_t (ascending: most confident first)
    sorted_records = sorted(records, key=lambda r: r.U_t)
    N = len(sorted_records)

    coverages = []
    risks = []

    for k in range(1, N + 1):
        subset = sorted_records[:k]
        cov = k / N
        err = 1.0 - (sum(1 for r in subset if r.success) / k)
        coverages.append(cov)
        risks.append(err)

    coverages = np.array(coverages)
    risks = np.array(risks)

    # RC-AUC: numerical trapezoidal integral
    rc_auc = float(np.trapz(risks, coverages)) if len(coverages) > 1 else 0.0
    return coverages, risks, rc_auc
