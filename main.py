"""
[NEW] [main_refactored.py](file:///c:/Users/lhkke/Documents/HullTactical/main_refactored.py)
Import dependencies from hull_tactical.
Check GPU availability.
Load data.
Run [run_all_models](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1416-1442).
Optionally run [run_pca_experiments](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py) based on config/args.
Save results.
"""
from typing import Dict
from data import load_data, time_train_test_split
from training import run_all_models, save_studies_to_disk
from config import path_str, TARGET_COL

df_raw = load_data(path_str.TRAIN_DIR)
df_cut_raw = df_raw[1006:]

y = df_cut_raw[TARGET_COL]


X_train, X_test, y_train, y_test = time_train_test_split(X, y)
studies = run_all_models(
    X_train,
    y_train,
    model_types=[

        "xgb"
    ],
    n_trials=100,
    n_splits=5,
    prune_features=False,
    prune_mode="stable",
    use_PCA=True,
)

out_dir = "optuna_results"
save_studies_to_disk(studies, "prune")