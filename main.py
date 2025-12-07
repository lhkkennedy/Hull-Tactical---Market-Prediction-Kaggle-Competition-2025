"""
[NEW] [main_refactored.py](file:///c:/Users/lhkke/Documents/HullTactical/main_refactored.py)
Import dependencies from hull_tactical.
Check GPU availability.
Load data.
Run [run_all_models](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1416-1442).
Optionally run [run_pca_experiments](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py) based on config/args.
Save results.
"""

from data import load_data, time_train_test_split
from training import run_all_models, save_studies_to_disk
from config import path_str, TARGET_COL

df_raw = load_data(path_str.TRAIN_DIR)
df_cut_raw = df_raw[1006:]

X = df_cut_raw.copy()
y = df_cut_raw[TARGET_COL]


X_train, X_test, y_train, y_test = time_train_test_split(X, y)
studies = run_all_models(
    X_train,
    y_train,
    model_types=[
        "ols", 
        "ridge", 
        "lasso", 
        "elastic", 
        "lgbm", 
        "xgb"
    ],
    n_trials=2,
    n_splits=3,
    prune_features=False,
)

out_dir = "optuna_results"
save_studies_to_disk(studies, "final")


FEATURE_RANKINGS: Dict[str, list] = {}

for mt in ["ols", "ridge", "lasso", "elastic", "lgbm", "xgb"]:
    if mt in ["ols", "ridge", "lasso", "elastic"]:
        FEATURE_RANKINGS[mt] = SortFeaturesByCorrElastic(X_train, y_train)
    elif mt in ["xgb", "lgbm"]:
        FEATURE_RANKINGS[mt] = SortFeaturesByImportanceXGB(X_train, y_train)
    else:
        FEATURE_RANKINGS[mt] = []


MODEL_TYPES = ["ols", "ridge", "lasso", "elastic", "lgbm", "xgb"]
LABELS = ["final_none", "pruned", "broad_tests"]  # Add/remove as needed

all_results = []

for label in LABELS:
    for model_type in MODEL_TYPES:
        print(f"Evaluating {model_type} ({label}) on holdout...")
        try:
            res = eval_best_config_on_holdout(
                model_type=model_type,
                label=label,
                X=X,
                y=y,
                feature_rankings=FEATURE_RANKINGS[model_type],
                test_frac=0.2,
                results_dir=RESULTS_DIR,
            )
            all_results.append(res)
        except FileNotFoundError:
            print(f"No trials file for {model_type} ({label}), skipping.")
        except Exception as e:
            print(f"[ERROR] {model_type} ({label}): {e}")


results_df = pd.DataFrame(all_results)
results_df.sort_values(["r2_holdout"], ascending=False, inplace=True)
print(results_df)

pivot_r2 = results_df.pivot(
    index="model_type",
    columns="label",
    values="r2_holdout",
)

pivot_r2.to_csv("pivot_r2.csv")
print(pivot_r2)
