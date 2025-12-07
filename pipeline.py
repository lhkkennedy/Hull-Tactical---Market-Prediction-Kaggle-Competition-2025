"""
[NEW] [hull_tactical/pipeline.py](file:///c:/Users/lhkke/Documents/HullTactical/hull_tactical/pipeline.py)
Move [create_pipeline](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1127-1141).
Move [create_pca_pipeline](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py) (New).
Move [get_final_estimator](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1143-1155).
Move FEATURE_LEVEL_CONFIGS.
"""

from sklearn.pipeline import Pipeline
from features import (
    BaseCleaner,
    TargetLagBuilder,
    TargetMomentumFeatures,
    CrossSectionalFeatureBuilder,
    VolatilityIndicators,
    dropExcludedCols,
    MissingValueImputer,
    PrecomputedTopKSelector,
    PCATransformer
)
from config import TARGET_COL, EXCLUDED_COLS

def create_pipeline(model, selector: PrecomputedTopKSelector | None = None):
    steps = [
        ("base_cleaner", BaseCleaner()),
        ("target_lag", TargetLagBuilder(target_col=TARGET_COL, lag_col="lagged_target", drop_target=True)),
        ("momentum", TargetMomentumFeatures()),
        ("features", CrossSectionalFeatureBuilder(sentinel=0.0)),
        ("vol", VolatilityIndicators()),
        ("drop", dropExcludedCols(cols_to_drop=list(EXCLUDED_COLS))),
        ("imputer", MissingValueImputer(sentinel=0.0)),
    ]
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", model))
    return Pipeline(steps)

def create_pca_pipeline(
    model,
    selector: PrecomputedTopKSelector | None = None,
    pca_transformer: PCATransformer | None = None,
):
    steps = [
        ("base_cleaner", BaseCleaner()),
        ("target_lag", TargetLagBuilder(target_col=TARGET_COL, lag_col="lagged_target", drop_target=True)),
        ("momentum", TargetMomentumFeatures()),
        ("features", CrossSectionalFeatureBuilder(sentinel=0.0)),
        ("vol", VolatilityIndicators()),
        ("drop", dropExcludedCols(cols_to_drop=list(EXCLUDED_COLS))),
        ("imputer", MissingValueImputer()),
    ]
    if selector is not None:
        steps.append(("selector", selector))
    if pca_transformer is not None:
        steps.append(("pca", pca_transformer))
    steps.append(("model", model))
    return Pipeline(steps)

def get_final_estimator(pipe_or_est):
    """
    Drill down through (nested) Pipelines to get the actual final estimator.
    """
    est = pipe_or_est

    # unwrap outer pipeline(s)
    while isinstance(est, Pipeline):
        # last step of the pipeline
        _, est = list(est.named_steps.items())[-1]

    return est