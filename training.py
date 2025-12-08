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
from pipeline import create_pipeline, get_final_estimator, create_pca_training_pipeline
from features import PrecomputedTopKSelector
from experiments import prepare_pca_metadata

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

def run_optuna_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    n_trials: int,
    n_splits: int,
    direction: str,
    prune_features: bool = False,
    use_PCA: bool = False,
    pca_mode: str | None = None,
    pca_meta: dict | None = None,
    feat_level: str = "none",
    label: str = "final",
    feature_ranking: list[str] | None = None,
) -> optuna.Study:
    """
    Run Optuna for a given model_type (optionally with PCA + feature pruning).
    """

    def objective(trial: optuna.Trial) -> float:
        # 1) build model from this trial's hyperparameters
        model = make_model(trial, model_type=model_type)

        # 2) optional top-k selector
        selector = None
        if prune_features and feature_ranking is not None:
            k_max = min(500, len(feature_ranking))
            k = trial.suggest_int("selector__k", 1, k_max)
            trial.set_user_attr("top_k_features", k)
            selector = PrecomputedTopKSelector(
                feature_ranking=feature_ranking,
                k=k,
                verbose=True,
            )

        # 3) build pipeline
        if use_PCA:
            if pca_mode is None or pca_meta is None:
                raise ValueError("use_PCA=True but pca_mode or pca_meta is None")

            pipe = create_pca_training_pipeline(
                model=model,
                selector=selector,
                pca_mode=pca_mode,
                pca_meta=pca_meta,
                feature_level="medium",  # PCA runs fixed at 'medium'
            )
        else:
            pipe = create_pipeline(model=model, selector=selector)
            pipe.set_params(**FEATURE_LEVEL_CONFIGS[feat_level])

        # 4) store metadata
        trial.set_user_attr("model_type", model_type)
        trial.set_user_attr("label", label)
        trial.set_user_attr("feat_level", feat_level)
        trial.set_user_attr("use_PCA", use_PCA)
        trial.set_user_attr("pca_mode", pca_mode)

        # 5) time-series CV
        scores = ts_cross_val_score(pipe, X, y, n_splits=n_splits)
        return float(np.mean(scores))

    study_name = f"{model_type}_study_{label}"
    study = optuna.create_study(direction=direction, study_name=study_name, sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials)
    return study


PCA_MODES = ["pca_only", "pca_hybrid", "pca_block"]

def run_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_types: list[str],
    n_trials: int = 50,
    n_splits: int = 3,
    direction: str = "maximize",
    prune_features: bool = False,
    use_PCA: bool = False,
    pca_modes: list[str] | None = None,
) -> dict[str, optuna.Study]:
    studies: dict[str, optuna.Study] = {}

    if use_PCA:
        # Prepare PCA metadata once
        pca_meta = prepare_pca_metadata(X, y)
        modes = pca_modes or PCA_MODES

        for model_type in model_types:
            for pca_mode in modes:
                print(f"\n=== Running Optuna for {model_type} | {pca_mode} (PCA) ===")
                study = run_optuna_for_model(
                    X=X,
                    y=y,
                    model_type=model_type,
                    n_trials=n_trials,
                    n_splits=n_splits,
                    direction=direction,
                    prune_features=prune_features,
                    use_PCA=True,
                    pca_mode=pca_mode,
                    pca_meta=pca_meta,
                    feat_level="extensive",
                )
                studies[(model_type, pca_mode)] = study

    else:
        # your existing non-PCA logic
        for model_type in model_types:
            if not prune_features:
                for feat_level in FEATURE_LEVEL_CONFIGS.keys():
                    print(f"\n=== Running Optuna for {model_type} | {feat_level} ===")
                    study = run_optuna_for_model(
                        X=X,
                        y=y,
                        model_type=model_type,
                        n_trials=n_trials,
                        n_splits=n_splits,
                        direction=direction,
                        prune_features=prune_features,
                        use_PCA=False,
                        pca_mode=None,
                        pca_meta=None,
                        feat_level=feat_level,
                    )
                    studies[(model_type, feat_level)] = study
            else:
                feat_level = "extensive"
                print(f"\n=== Running Optuna for {model_type} | {feat_level} ===")
                study = run_optuna_for_model(
                    X=X,
                    y=y,
                    model_type=model_type,
                    n_trials=n_trials,
                    n_splits=n_splits,
                    direction=direction,
                    prune_features=prune_features,
                    use_PCA=False,
                    pca_mode=None,
                    pca_meta=None,
                    feat_level=feat_level,
                )
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