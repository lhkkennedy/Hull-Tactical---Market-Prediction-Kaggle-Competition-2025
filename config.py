"""
[NEW] [hull_tactical/config.py](file:///c:/Users/lhkke/Documents/HullTactical/hull_tactical/config.py)
Move [path_str](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#36-43) dataclass.
Move constants: EXCLUDED_COLS, DATE_COL, TARGET_COL, PREFIX_BUCKETS.
Move USE_GPU.
"""

from dataclasses import dataclass

USE_GPU = True

@dataclass(frozen=True)
class path_str:
    TRAIN_DIR: str = "kaggle/input/hull-tactical-market-prediction/train.csv"
    TEST_DIR: str = "/kaggle/input/hull-tactical-market-prediction/test.cv"
    MODELS_DIR: str = "/kaggle/working/models"
    OOF_DIR: str = "/kaggle/working/oof"
    METRICS_CSV: str = "/kaggle/working/metrics.csv"

EXCLUDED_COLS = {
    'date_id',
    'forward_returns',
    'risk_free_rate',
    'market_forward_excess_returns',
}

DATE_COL = 'date_id'

TARGET_COL = 'market_forward_excess_returns'

PREFIX_BUCKETS = {
    "M": "Market",
    "P": "Price",
    "V": "Volatility",
    "S": "Sentiment",
    "D": "Dummy",
    "I": "Interest",
    "E": "Macro",
}

FEATURE_LEVEL_CONFIGS = {
    "none": dict(
        features__enabled=False,
        momentum__enabled=False,
        vol__enabled=False,
    ),
    "simple": dict(
        features__enabled=False,
        momentum__enabled=True,
        vol__enabled=False,
    ),
    "simple2": dict(
        features__enabled=False,
        momentum__enabled=True,
        vol__enabled=True,
        vol__drop_regime_labels=True,
        vol__add_interactions=True,
    ),
    "medium": dict(
        features__enabled=True,
        features__windows=(5, 10, 21, 63),
        features__lags=(1, 2, 3, 5, 10, 21),
        momentum__enabled=True,
        vol__enabled=False,
    ),
    "advanced": dict(
        features__enabled=True,
        features__windows=(5, 10, 21, 63),
        features__lags=(1, 2, 3, 5, 10, 21),
        momentum__enabled=True,
        vol__enabled=True,
        vol__drop_regime_labels=True,
        vol__add_interactions=True,
    ),
    "extensive": dict(
        features__enabled=True,
        features__windows=(5, 10, 21, 63, 126, 252),
        features__lags=(1, 2, 3, 5, 10, 21, 63, 126),
        momentum__enabled=True,
        vol__enabled=True,
        vol__drop_regime_labels=True,
        vol__add_interactions=True,
    ),
}