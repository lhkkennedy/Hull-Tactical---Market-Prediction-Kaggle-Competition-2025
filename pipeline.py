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
        ("features", CrossSectionalFeatureBuilder()),
        ("vol", VolatilityIndicators()),
        ("drop", dropExcludedCols(cols_to_drop=list(EXCLUDED_COLS))),
        ("imputer", MissingValueImputer(createMissingFlags=True)),
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
        ("features", CrossSectionalFeatureBuilder()),
        ("vol", VolatilityIndicators()),
        ("drop", dropExcludedCols(cols_to_drop=list(EXCLUDED_COLS))),
        ("imputer", MissingValueImputer(createMissingFlags=True)),
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

# pipeline.py

from sklearn.pipeline import Pipeline
from config import EXCLUDED_COLS, FEATURE_LEVEL_CONFIGS, TARGET_COL
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

def create_pca_training_pipeline(
    model,
    selector,
    pca_mode: str,
    pca_meta: dict,
    feature_level: str = "medium",
) -> Pipeline:
    """
    Build a training pipeline for a given PCA regime.

    pca_mode: "pca_only", "pca_hybrid", or "pca_block"
    pca_meta: dict from prepare_pca_metadata(...)
    """
    if pca_mode == "pca_only":
        # PCA on all post-preproc features
        pca_only = PCATransformer(
            columns=None,
            n_components=20,
            keep_other=False,
            prefix="PC",
            standardize=True,
        )
        pipe = create_pca_pipeline(
            model=model,
            selector=selector,
            pca_transformer=pca_only,
        )

    elif pca_mode == "pca_hybrid":
        # PCA on selected block, keep others raw
        pca_hybrid = PCATransformer(
            columns=pca_meta["hybrid_cols"],
            n_components=20,
            keep_other=True,
            prefix="PC_blk",
            standardize=True,
        )
        pipe = create_pca_pipeline(
            model=model,
            selector=selector,
            pca_transformer=pca_hybrid,
        )

    elif pca_mode == "pca_block":
        # Manually build multi-block PCA pipeline
        steps_base = [
            ("base_cleaner", BaseCleaner()),
            ("target_lag", TargetLagBuilder(
                target_col=TARGET_COL,
                lag_col="lagged_target",
                drop_target=True,
            )),
            ("momentum", TargetMomentumFeatures()),
            ("features", CrossSectionalFeatureBuilder(sentinel=0.0)),
            ("vol", VolatilityIndicators()),
            ("drop", dropExcludedCols(cols_to_drop=list(EXCLUDED_COLS))),
            ("imputer", MissingValueImputer(sentinel=0.0)),
        ]

        steps_pca_blocks = steps_base + [
            ("pca_market", PCATransformer(
                columns=pca_meta["market_cols"],
                n_components=5,
                keep_other=True,
                prefix="PC_M",
            )),
            ("pca_price", PCATransformer(
                columns=pca_meta["price_cols"],
                n_components=5,
                keep_other=True,
                prefix="PC_P",
            )),
            ("pca_vol", PCATransformer(
                columns=pca_meta["vol_cols"],
                n_components=3,
                keep_other=True,
                prefix="PC_V",
            )),
            ("pca_sent", PCATransformer(
                columns=pca_meta["sent_cols"],
                n_components=3,
                keep_other=True,
                prefix="PC_S",
            )),
            ("model", model),
        ]

        pipe = Pipeline(steps_pca_blocks)

    else:
        raise ValueError(f"Unknown pca_mode: {pca_mode}")

    # Use your "medium" preproc config by default
    pipe.set_params(**FEATURE_LEVEL_CONFIGS[feature_level])
    return pipe
