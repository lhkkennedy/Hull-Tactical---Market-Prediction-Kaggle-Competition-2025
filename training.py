"""
[NEW] [hull_tactical/training.py](file:///c:/Users/lhkke/Documents/HullTactical/hull_tactical/training.py)
Move [SortFeaturesByCorrElastic](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1157-1185), [SortFeaturesByImportanceXGB](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1186-1217).
Move [ts_cross_val_score](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1218-1227).
Move [optuna_objective](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1230-1328).
Move [run_optuna_for_model](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1330-1396).
Move [run_all_models](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1416-1442).
Move [eval_tuned_model_on_holdout](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1528-1572).
Move [save_studies_to_disk](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1602-1635) (if present/used).
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from typing import Literal
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from config import FEATURE_LEVEL_CONFIGS
from models import make_model, make_elasticnet, make_xgb
from pipeline import create_pipeline, get_final_estimator
from features import PrecomputedTopKSelector

def transform_through_preproc(pipe, X):
    """
    Transform X through all steps of the pipeline except the last one (the model),
    reusing the *already fitted* steps from `pipe`.
    """
    Xt = X
    # pipe.steps is a list of (name, transformer/estimator)
    for name, step in pipe.steps[:-1]:
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
    return Xt

def SortFeaturesByCorrElastic(X_train, y_train):
    pipe = create_pipeline(
        make_elasticnet(
            alpha=0.0009609238331494418,
            l1_ratio=0.5419274494337433,
        )
    )
    pipe.set_params(**FEATURE_LEVEL_CONFIGS["extensive"])
    pipe.fit(X_train, y_train)

    X_trans = transform_through_preproc(pipe, X_train)

    if isinstance(X_trans, pd.DataFrame):
        feature_names = np.array(X_trans.columns)
    else:
        feature_names = np.array([f"feat_{i}" for i in range(X_trans.shape[1])])
    final_est = get_final_estimator(pipe.named_steps["model"])
    coefs = np.asarray(final_est.coef_)

    idx_sorted = np.argsort(np.abs(coefs))[::-1]
    sorted_features = feature_names[idx_sorted]

    return list(sorted_features)


def SortFeaturesByImportanceXGB(X_train, y_train):
    pipe = create_pipeline(make_xgb(params={
        "max_depth": 8,
        "min_child_weight": 10,
        "subsample": 0.7112655772048546,
        "colsample_bytree": 0.5041689568795529,
        "learning_rate": 0.0011550749274408143,
        "n_estimators": 583,
        "reg_lambda": 0.3216770122073615,
        "reg_alpha": 0.26826781475439687,
    }))
    pipe.set_params(**FEATURE_LEVEL_CONFIGS["extensive"])
    pipe.fit(X_train, y_train)

    X_trans = transform_through_preproc(pipe, X_train)

    if isinstance(X_trans, pd.DataFrame):
        feature_names = np.array(X_trans.columns)
    else:
        feature_names = np.array([f"feat_{i}" for i in range(X_trans.shape[1])])

    final_est = get_final_estimator(pipe.named_steps["model"])
    importances = np.asarray(final_est.feature_importances_)

    idx_sorted = np.argsort(importances)[::-1]
    sorted_features = feature_names[idx_sorted]

    return list(sorted_features)

def ts_cross_val_score(pipe, X, y, n_splits=3, use_early_stopping=False, es_rounds=50):
    """
    Plain time-series CV using sklearn's cross_val_score.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring="r2", n_jobs=-1)
    return scores

def optuna_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgb",
    n_splits: int = 3,
    feat_level: str = None,
    prune_features: bool = False,
    sorted_features: bool = False,
) -> float:
    """
    Single Optuna objective:
    - choose feature level (none / simple / medium / extensive)
    - configure preproc toggles
    - build model via make_model(...)
    - wrap everything in create_pipeline(...)
    - evaluate with time-series CV
    """

    X_local = X.copy()
    
    trial.set_user_attr("feature_level", feat_level)


    # 2) build base model with its own hyperparams
    model = make_model(trial, model_type=model_type)


    selector = None
    if prune_features and sorted_features is not None:
        k_max = min(500, len(sorted_features))
        k = trial.suggest_int("selector__k", 1, k_max)
        trial.set_user_attr("top_k_features", k)
        selector = PrecomputedTopKSelector(feature_ranking=sorted_features, k=k, verbose=True)


    # 3) build full pipeline skeleton
    pipe = create_pipeline(model, selector=selector)


    # 4) apply coarse configuration for this feature level
    pipe.set_params(**FEATURE_LEVEL_CONFIGS[feat_level])

    # # 5) fine-grained preproc toggles (example, expand as needed)
    # if level in ("medium", "extensive"):
    #     # Momentum feature toggles
    #     pipe.set_params(
    #         momentum__use_mean=trial.suggest_categorical(
    #             "momentum__use_mean", [True, False]
    #         ),
    #         momentum__use_std=trial.suggest_categorical(
    #             "momentum__use_std", [True, False]
    #         ),
    #     )
    # ...

    # 6) evaluate with time-series CV
    scores = ts_cross_val_score(
        pipe,
        X_local,
        y,
        n_splits=n_splits,
    )
    # scores should already be "higher is better" (R², Sharpe, etc.)
    return float(np.mean(scores))

def run_optuna_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: Literal[
        "ols", "ridge", "lasso", "elastic", "lgbm", "xgb"
    ] = "xgb",
    n_trials: int = 50,
    n_splits: int = 3,
    direction: str = "maximize",
    storage: str | None = None,
    feat_level: str | None = None,
    prune_features: bool = False,
) -> optuna.Study:
    prune_str = "prune" if prune_features else "no_prune"
    study_name = f"{model_type}_study_{feat_level}_{prune_str}"

    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
    )

    # precompute sorted features in RAW space
    sorted_features = None
    if prune_features:
        if model_type in ["elastic", "lasso", "ridge", "ols"]:
            sorted_features = SortFeaturesByCorrElastic(X, y)
        elif model_type in ["xgb", "lgbm"]:
            sorted_features = SortFeaturesByImportanceXGB(X, y)

    def _objective(trial: optuna.Trial) -> float:
        return optuna_objective(
            trial=trial,
            X=X,
            y=y,
            model_type=model_type,
            n_splits=n_splits,
            feat_level=feat_level,
            prune_features=prune_features,
            sorted_features=sorted_features,
        )

    study.optimize(
        _objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"[{model_type} | feat_level={feat_level}] Best value: {study.best_value}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        print(study)

    return study

def run_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_types: list[str],
    n_trials: int = 50,
    n_splits: int = 3,
    direction: str = "maximize",
    fixed_features: bool = False,   # currently unused, keep if you plan to resurrect it
    prune_features: bool = False,
) -> dict[tuple[str, str], optuna.Study]:
    """
    Returns dict with keys (model_type, feat_level) -> Study
    """
    studies: dict[tuple[str, str], optuna.Study] = {}

    for model_type in model_types:
        for feat_level in FEATURE_LEVEL_CONFIGS.keys():
            print(f"\n=== Running Optuna for {model_type} | {feat_level} ===")
            study = run_optuna_for_model(
                X=X,
                y=y,
                model_type=model_type,
                n_trials=n_trials,
                n_splits=n_splits,
                direction=direction,
                feat_level=feat_level,
                prune_features=prune_features,
            )
            # key is now a tuple, not a mashed string
            studies[(model_type, feat_level)] = study

    return studies

def save_studies_to_disk(
    studies: dict[tuple[str, str], optuna.Study],
    label: str,
    out_dir: str = "optuna_results",
) -> None:
    """
    Save each study's trials dataframe and a summary CSV.

    - One CSV per (model_type, feature_level):
        <model_type>_<feature_level>_<label>_trials.csv
    - One summary CSV: summary_<label>.csv
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for (model_type, feature_level), study in studies.items():
        df = study.trials_dataframe()

        # Per-model+feature_level trials file
        trials_filename = f"{model_type}_{feature_level}_{label}_trials.csv"
        df.to_csv(out_path / trials_filename, index=False)

        # Row for summary file
        row = {
            "model_type": model_type,
            "feature_level": feature_level,
            "label": label,
            "best_value": study.best_value,
        }
        for k, v in study.best_params.items():
            row[f"best_{k}"] = v
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path / f"summary_{label}.csv", index=False)