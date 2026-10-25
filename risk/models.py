"""Risk prediction models for grasp execution failure.

Four risk models fit on the fusion split:
1. score-only: Logistic regression on (1 - s*)
2. +dispersion: Logistic regression on [1 - s*, largest_cluster_mass, normalized_entropy]
3. +cross-view: Gradient boosting on [1 - s*, norm_view_idx, norm_dist_prev, score_gap]
4. all-features: Gradient boosting on all features
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


class RiskModelSuite:
    """Trains and evaluates the 4 comparative risk models on episode data."""

    def __init__(self):
        self.models = {
            "score_only": LogisticRegression(penalty=None, solver="lbfgs"),
            "dispersion": LogisticRegression(penalty=None, solver="lbfgs"),
            "cross_view": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
            "all_features": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        }
        self.is_fitted = False

    def _extract_feature_subsets(self, X_all: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract input features for each model.
        
        Features format:
        0: 1 - s*
        1: largest_cluster_mass
        2: normalized_entropy
        3: norm_view_idx
        4: norm_dist_prev
        5: score_gap
        """
        return {
            "score_only": X_all[:, [0]],
            "dispersion": X_all[:, [0, 1, 2]],
            "cross_view": X_all[:, [0, 3, 4, 5]],
            "all_features": X_all,
        }

    def fit(self, features: np.ndarray, outcomes: np.ndarray):
        """Fit all 4 models on the fusion split.
        
        Args:
            features: (N_episodes, T, 6) or (N_episodes * T, 6)
            outcomes: (N_episodes, T) or (N_episodes * T,) binary outcomes (1=success, 0=failure)
        """
        # Target is failure: y = 1 if failed (outcome == 0), 0 if succeeded (outcome == 1)
        X_flat = features.reshape(-1, 6)
        y_flat = (outcomes.reshape(-1) == 0).astype(int)

        # In case all labels are same class in small smoke tests
        if len(np.unique(y_flat)) < 2:
            # Add synthetic opposite label
            X_flat = np.vstack([X_flat, np.zeros((1, 6)), np.ones((1, 6))])
            y_flat = np.concatenate([y_flat, [0, 1]])

        feat_subsets = self._extract_feature_subsets(X_flat)
        for name, model in self.models.items():
            model.fit(feat_subsets[name], y_flat)

        self.is_fitted = True

    def predict_risk(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict failure probability rho in [0, 1] for all models.
        
        Args:
            features: (N, T, 6) or (N, 6)
        Returns:
            dict of arrays of risk estimates rho with shape (N, T) or (N,)
        """
        orig_shape = features.shape[:-1]
        X_flat = features.reshape(-1, 6)
        feat_subsets = self._extract_feature_subsets(X_flat)

        risks = {}
        for name, model in self.models.items():
            # Failure probability is class 1
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feat_subsets[name])
                if probs.shape[1] == 2:
                    p_fail = probs[:, 1]
                else:
                    p_fail = probs[:, 0]
            else:
                p_fail = X_flat[:, 0]  # Fallback 1 - s*

            risks[name] = p_fail.reshape(orig_shape)

        return risks
