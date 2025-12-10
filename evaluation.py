from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from models import (
    make_ols,
    make_ridge,
    make_lasso,
    make_elasticnet,
    make_xgb,
    make_lgbm,
)
from pipeline import create_pipeline
from features import PrecomputedTopKSelector
from config import FEATURE_LEVEL_CONFIGS
from data import time_train_test_split


# -------------------------------------------------------------------
# 1. Best-trial loader
# -------------------------------------------------------------------

def get_best_trial_info(
    model_type: str,
    label: str,
    results_dir: str,
    feat_level: str,
) -> Tuple[Dict[str, Any], float]:
    """
    Load the trials CSV for a given (model_type, feat_level, label),
    pick the best row (max value), and return:
      - params dict (without 'params_' prefix)
      - best CV value
    """
    path = Path(results_dir) / f"{model_type}_{feat_level}_{label}_trials.csv"
    print(f"Loading trials from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"No trials file found: {path}")

    df = pd.read_csv(path)

    if "value" not in df.columns:
        raise ValueError(f"'value' column not found in {path}")

    best_idx = df["value"].idxmax()
    row = df.loc[best_idx]

    # All Optuna param columns look like 'params_xxx'
    param_cols = [c for c in df.columns if c.startswith("params_")]
    params = {c.replace("params_", ""): row[c] for c in param_cols}

    best_value = float(row["value"])
    return params, best_value


# -------------------------------------------------------------------
# 2. Model / pipeline builders
# -------------------------------------------------------------------

def build_estimator_from_params(
    model_type: str,
    params: Dict[str, Any],
):
    """
    Turn Optuna params into a base estimator for each model_type.
    """
    if model_type == "ols":
        est = make_ols()
    elif model_type == "ridge":
        est = make_ridge(alpha=params.get("ridge_alpha", 1.0))
    elif model_type == "lasso":
        est = make_lasso(alpha=params.get("lasso_alpha", 1e-3))
    elif model_type == "elastic":
        est = make_elasticnet(
            alpha=params.get("enet_alpha", 1e-3),
            l1_ratio=params.get("enet_l1_ratio", 0.5),
        )
    elif model_type == "xgb":
        est = make_xgb(params=params)
    elif model_type == "lgbm":
        est = make_lgbm(params=params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return est


def build_pipeline_from_best_trial(
    model_type: str,
    label: str,
    feature_rankings: list | None,
    results_dir: str,
    feat_level: str,
):
    """
    - Load best trial params for the given (model_type, feat_level, label)
    - Build base estimator
    - Optionally build selector (if selector__k is present in params)
    - Assemble pipeline in the same structure as training
    - Apply FEATURE_LEVEL_CONFIGS[feat_level]
    """
    params, best_cv_value = get_best_trial_info(
        model_type=model_type,
        label=label,
        results_dir=results_dir,
        feat_level=feat_level,
    )

    est = build_estimator_from_params(model_type, params)

    # If tuning used a selector (prune_features=True), Optuna will have selector__k
    selector = None
    if "selector__k" in params and feature_rankings is not None:
        k = int(params["selector__k"])
        selector = PrecomputedTopKSelector(
            feature_ranking=feature_rankings,
            k=k,
            # keep these consistent with training:
            decorrelate=True,
            corr_thresh=0.9,
        )

    pipe = create_pipeline(model=est, selector=selector)

    # apply coarse feature-level config (none/simple/simple2/medium/advanced/extensive)
    cfg = FEATURE_LEVEL_CONFIGS.get(feat_level, {})
    if cfg:
        pipe.set_params(**cfg)

    feature_level = feat_level
    return pipe, params, feature_level, best_cv_value


# -------------------------------------------------------------------
# 3. Evaluation on holdout
# -------------------------------------------------------------------

def eval_best_config_on_holdout(
    model_type: str,
    label: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_rankings: list | None,
    results_dir: str,
    feat_level: str,
    test_frac: float = 0.2,
) -> Dict[str, Any]:
    """
    - Rebuild tuned pipeline (with selector if selector__k was tuned)
    - Chronological split
    - Train on train, evaluate on holdout
    """
    pipe, params, feature_level, best_cv_value = build_pipeline_from_best_trial(
        model_type=model_type,
        label=label,
        feature_rankings=feature_rankings,
        results_dir=results_dir,
        feat_level=feat_level,
    )

    # chronological train/test split for HOLDOUT
    X_train, X_test, y_train, y_test = time_train_test_split(
        X, y, test_frac=test_frac
    )

    # DEBUG sanity check: columns should match
    if set(X_train.columns) != set(X_test.columns):
        diff_train = set(X_train.columns) - set(X_test.columns)
        diff_test = set(X_test.columns) - set(X_train.columns)
        print("[WARN] Column mismatch between train and test.")
        print("  Only in train:", diff_train)
        print("  Only in test :", diff_test)

    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    r2_tr  = r2_score(y_train, y_pred_train)
    r2_te  = r2_score(y_test, y_pred_test)
    mse_tr = mean_squared_error(y_train, y_pred_train)
    mse_te = mean_squared_error(y_test, y_pred_test)

    selector_k = params.get("selector__k", np.nan)

    return {
        "model_type": model_type,
        "label": label,
        "feature_level": feature_level,
        "best_cv_value": best_cv_value,
        "r2_insample": r2_tr,
        "r2_holdout": r2_te,
        "mse_insample": mse_tr,
        "mse_holdout": mse_te,
        "selector_k": selector_k,
    }
