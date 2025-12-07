"""
[NEW] [hull_tactical/data.py](file:///c:/Users/lhkke/Documents/HullTactical/hull_tactical/data.py)
Move [load_data](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#66-68).
Move [time_train_test_split](file:///c:/Users/lhkke/Documents/HullTactical/final_training_script.py#1443-1454).
"""

from typing import Optional, Tuple
import pandas as pd
from config import path_str, DATE_COL


def load_data(path: str) -> pd.DataFrame:
    # Note: path_str is a class with constants, so path argument here should probably be a string.
    # However, if the user calls load_data(path_str.TRAIN_DIR), it IS a string.
    # The type hint `path: path_str` in the original was probably wrong or meant `str`.
    # I'll change type hint to `str` to be safe, or Import path_str and use `str`.
    return pd.read_csv(path)

def time_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    test_frac: float = 0.2, # Added to match typical usage if sizes not provided
    date_col: str = DATE_COL,
    shuffle: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets based on time.
    """
    if shuffle:
        raise ValueError("Shuffling is not supported for time-based splits.")

    if train_size is not None and test_size is not None:
        train_end = train_size
        test_end = train_end + test_size
        
        X_train = X.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:test_end]
    else:
        # Fallback to test_frac logic if sizes are not explicit (typical in simple scripts)
        n = len(X)
        split = int(n * (1 - test_frac))
        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test