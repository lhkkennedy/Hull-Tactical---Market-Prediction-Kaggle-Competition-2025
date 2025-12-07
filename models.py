"""
[NEW] [hull_tactical/models.py](file:///c:/Users/lhkke/Documents/HullTactical/hull_tactical/models.py)
Move model factory functions:
[make_ols](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#904-909), [make_ridge](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#910-918), [make_lasso](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#919-928), [make_elasticnet](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#929-939)
[make_lgbm](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#940-953), [make_xgb](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#954-972)
[make_model](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#973-1025) (Optuna trial integration)
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from config import USE_GPU

def make_ols():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

def make_ridge(alpha=0.1):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(
            alpha=alpha,
            random_state=42
        ))
    ])

def make_lasso(alpha=1e-3):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(
            alpha=alpha,
            random_state=42,
            max_iter=10000
        ))
    ])

def make_elasticnet(alpha=1e-3, l1_ratio=0.5):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
            max_iter=10000
        ))
    ])

def make_lgbm(params=None, use_gpu: bool = USE_GPU):
    params = params or {}
    base = dict(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1,
    )
    base.update(params)
    return LGBMRegressor(**base)

def make_xgb(params=None, use_gpu: bool = USE_GPU):
    params = params or {}
    base = dict(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        verbosity=0,
        eval_metric="rmse",
    )
    if use_gpu:
        base["device"] = "cuda"
        
    base.update(params)
    return XGBRegressor(**base)

def make_model(trial, model_type: str):
    """
    Build the *final* estimator (possibly a Pipeline with scaler/PCA)
    based on model_type and Optuna trial.
    """

    if model_type == "ols":
        return make_ols()

    if model_type == "ridge":
        alpha = trial.suggest_float("ridge_alpha", 1e-4, 1e3, log=True)
        return make_ridge(alpha=alpha)

    if model_type == "lasso":
        alpha = trial.suggest_float("lasso_alpha", 1e-4, 10.0, log=True)
        return make_lasso(alpha=alpha)

    if model_type == "elastic":
        # Adjusted range: 10.0 is too high for standardized data and causes model collapse
        alpha = trial.suggest_float("enet_alpha", 1e-5, 0.1, log=True)
        l1_ratio = trial.suggest_float("enet_l1_ratio", 0.0, 1.0)
        return make_elasticnet(alpha=alpha, l1_ratio=l1_ratio)

    if model_type == "lgbm":
        params = {
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 64, 128),
            "max_depth": trial.suggest_int("lgbm_max_depth", 3, 12),
            "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.005, 0.1, log=True),
            "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 10, 100),
            "subsample": trial.suggest_float("lgbm_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.5, 1.0),
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 4000),
            "reg_alpha": trial.suggest_float("lgbm_reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 0.1, 20.0, log=True),
        }
        return make_lgbm(params=params)

    if model_type == "xgb":
        # plain XGB, no scaler
        params = {
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 20),
            "subsample": trial.suggest_float("xgb_subsample", 0.3, 0.9),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.3, 0.9),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.001, 0.05, log=True),
            "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 4000),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.1, 20.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 5.0),
        }
        return make_xgb(params=params)

    raise ValueError(f"Unknown model_type: {model_type}")
