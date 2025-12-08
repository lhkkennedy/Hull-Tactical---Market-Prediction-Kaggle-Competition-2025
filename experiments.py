"""
[NEW] [hull_tactical/experiments.py](file:///c:/Users/lhkke/Documents/HullTactical/hull_tactical/experiments.py)
Move PCA Experiment Logic:
[infer_pca_block_columns](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py)
[columns_with_prefix](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py)
[run_pca_experiments](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py)
"""

import pandas as pd
from data import time_train_test_split
from pipeline import create_pipeline, create_pca_pipeline
from models import make_ridge
from config import EXCLUDED_COLS, FEATURE_LEVEL_CONFIGS, TARGET_COL, DATE_COL
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from features import (
    BaseCleaner,
    TargetLagBuilder,
    TargetMomentumFeatures,
    CrossSectionalFeatureBuilder,
    VolatilityIndicators,
    dropExcludedCols,
    MissingValueImputer,
    PCATransformer,
)
def infer_pca_block_columns(X_sample: pd.DataFrame) -> list[str]:
    """
    Select columns for 'PCA + raw' hybrid approach.
    Excluded: EXCLUDED_COLS, regimes, lagged targets, dummies.
    """
    cols = []
    for c in X_sample.columns:
        if c in EXCLUDED_COLS:
            continue
        if c.startswith(("VI_regime_", "regime_", "lagged_", "D")):
            continue
        cols.append(c)
    return cols

def columns_with_prefix(X_sample: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    """
    Select columns matching specific prefixes (e.g. ('M',) for Market).
    """
    cols = []
    for c in X_sample.columns:
        if any(c.startswith(p) for p in prefixes):
            cols.append(c)
    return cols

def run_pca_experiments(X: pd.DataFrame, y: pd.Series):
    print("\n========================================")
    print("Running PCA Experiments")
    print("========================================")
    
    # 1. Split
    X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_frac=0.2)
    
    # Helper to get feature names / columns from a 'medium' run
    print("...running feature generation on train set to identify columns...")
    tmp_pipe = create_pipeline(make_ridge())
    tmp_pipe.set_params(**FEATURE_LEVEL_CONFIGS["medium"])
    
    # Fit preprocessor only (slice up to 'model')
    # steps: base_cleaner, target_lag, momentum, features, vol, drop, imputer, model
    # We want everything EXCEPT model
    preproc_steps = tmp_pipe.steps[:-1]
    preproc_pipe = Pipeline(preproc_steps)
    
    preproc_pipe.fit(X_train, y_train)
    X_train_feat = X_train
    for name, step in preproc_pipe.steps:
        if hasattr(step, "transform"):
            X_train_feat = step.transform(X_train_feat)
    
    # ---------------------------------------------------------
    # Option 1: PCA-only regression
    # ---------------------------------------------------------
    print("\n[PCA Option 1] PCA-only (Ridge)")
    ridge_model1 = make_ridge(alpha=0.1)
    
    pca_only = PCATransformer(
        columns=None,          # all columns
        n_components=20,       
        keep_other=False,      
        prefix="PC",
        standardize=True,
    )
    
    pipe_pca_only = create_pca_pipeline(
        model=ridge_model1,
        selector=None, 
        pca_transformer=pca_only,
    )
    pipe_pca_only.set_params(**FEATURE_LEVEL_CONFIGS["medium"])
    
    pipe_pca_only.fit(X_train, y_train)
    y_pred1 = pipe_pca_only.predict(X_test)
    r2_1 = r2_score(y_test, y_pred1)
    print(f"R2 PCA-only: {r2_1:.5f}")

    # ---------------------------------------------------------
    # Option 2: PCA + raw features
    # ---------------------------------------------------------
    print("\n[PCA Option 2] Hybrid PCA + Raw (Ridge)")
    
    pca_cols_opt2 = infer_pca_block_columns(X_train_feat)
    print(f"Hybrid PCA: compressing {len(pca_cols_opt2)} columns, keeping others raw.")

    ridge_model2 = make_ridge(alpha=0.1)
    
    pca_hybrid = PCATransformer(
        columns=pca_cols_opt2,
        n_components=20,
        keep_other=True,
        prefix="PC_blk",
        standardize=True,
    )
    
    pipe_pca_hybrid = create_pca_pipeline(
        model=ridge_model2,
        selector=None,
        pca_transformer=pca_hybrid,
    )
    pipe_pca_hybrid.set_params(**FEATURE_LEVEL_CONFIGS["medium"]) 
    
    pipe_pca_hybrid.fit(X_train, y_train)
    y_pred2 = pipe_pca_hybrid.predict(X_test)
    r2_2 = r2_score(y_test, y_pred2)
    print(f"R2 PCA-hybrid: {r2_2:.5f}")

    # ---------------------------------------------------------
    # Option 3: Block / Group PCA
    # ---------------------------------------------------------
    print("\n[PCA Option 3] Block PCA (Ridge)")
    
    market_cols = columns_with_prefix(X_train_feat, ("M",))
    price_cols  = columns_with_prefix(X_train_feat, ("P",))
    vol_cols    = columns_with_prefix(X_train_feat, ("V",))
    sent_cols   = columns_with_prefix(X_train_feat, ("S",))
    
    print(f"Blocks: M={len(market_cols)}, P={len(price_cols)}, V={len(vol_cols)}, S={len(sent_cols)}")
    
    ridge_model3 = make_ridge(alpha=0.1)
    
    # Manually constructing pipeline to allow multiple PCAs
    steps_base = [
        ("base_cleaner", BaseCleaner()),
        ("target_lag", TargetLagBuilder(target_col=TARGET_COL, lag_col="lagged_target", drop_target=True)),
        ("momentum", TargetMomentumFeatures()),
        ("features", CrossSectionalFeatureBuilder(sentinel=0.0)),
        ("vol", VolatilityIndicators()),
        ("drop", dropExcludedCols(cols_to_drop=list(EXCLUDED_COLS))),
        ("imputer", MissingValueImputer(sentinel=0.0)),
    ]
    
    steps_pca_blocks = steps_base + [
        ("pca_market", PCATransformer(columns=market_cols, n_components=5, keep_other=True, prefix="PC_M")),
        ("pca_price",  PCATransformer(columns=price_cols,  n_components=5, keep_other=True, prefix="PC_P")),
        ("pca_vol",    PCATransformer(columns=vol_cols,    n_components=3, keep_other=True, prefix="PC_V")),
        ("pca_sent",   PCATransformer(columns=sent_cols,   n_components=3, keep_other=True, prefix="PC_S")),
        ("model", ridge_model3),
    ]
    
    pipe_block_pca = Pipeline(steps_pca_blocks)
    pipe_block_pca.set_params(**FEATURE_LEVEL_CONFIGS["medium"])
    
    pipe_block_pca.fit(X_train, y_train)
    y_pred3 = pipe_block_pca.predict(X_test)
    r2_3 = r2_score(y_test, y_pred3)
    print(f"R2 PCA-Block: {r2_3:.5f}")


def prepare_pca_metadata(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Run the preprocessing stack once on a train split and
    infer which columns to feed into each PCA variant.
    Returns a dict with all the column lists needed.
    """
    # same split logic you use elsewhere
    X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_frac=0.2)

    # Build a "medium" pipeline and strip off the model
    tmp_pipe = create_pipeline(make_ridge())
    tmp_pipe.set_params(**FEATURE_LEVEL_CONFIGS["medium"])

    preproc_steps = tmp_pipe.steps[:-1]
    preproc_pipe = Pipeline(preproc_steps)

    preproc_pipe.fit(X_train, y_train)

    # Manually transform through the fitted preprocessing steps
    X_train_feat = X_train
    for name, step in preproc_pipe.steps:
        if hasattr(step, "transform"):
            X_train_feat = step.transform(X_train_feat)

    if not isinstance(X_train_feat, pd.DataFrame):
        X_train_feat = pd.DataFrame(X_train_feat, index=X_train.index)

    # Hybrid PCA columns (compress these, keep others raw)
    hybrid_cols = infer_pca_block_columns(X_train_feat)

    # Block PCA groups
    market_cols = columns_with_prefix(X_train_feat, ("M",))
    price_cols  = columns_with_prefix(X_train_feat, ("P",))
    vol_cols    = columns_with_prefix(X_train_feat, ("V",))
    sent_cols   = columns_with_prefix(X_train_feat, ("S",))

    return {
        "hybrid_cols": hybrid_cols,
        "market_cols": market_cols,
        "price_cols":  price_cols,
        "vol_cols":    vol_cols,
        "sent_cols":   sent_cols,
    }
