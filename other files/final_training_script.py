import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Literal

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
from optuna.samplers import TPESampler

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone


from tqdm.notebook import tqdm
from itertools import product

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

def load_data(path: path_str) -> pd.DataFrame:
    return pd.read_csv(path)

class BaseCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self
    def transform(self, X: pd.DataFrame):
        df= X.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

class TargetLagBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, lag_col="lagged_target", drop_target=True):
        self.target_col = target_col
        self.lag_col = lag_col
        self.drop_target = drop_target

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in DataFrame.")

        s = pd.to_numeric(df[self.target_col], errors="coerce")
        df[self.lag_col] = s.shift(1)

        if self.drop_target:
            df = df.drop(columns=[self.target_col], errors="ignore")

        return df


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in a DataFrame.

    Parameters
    ----------
    sentinel : float, optional
        Value to use as a sentinel for missing values, by default 0.0
    createMissingFlags : bool, optional
        Whether to create missing flags, by default False
        
    """

    def __init__(
        self,
        sentinel:   float = 0.0,
        createMissingFlags: bool = False,
    ) -> None:
        self.sentinel = sentinel
        self.createMissingFlags = createMissingFlags

    # -------------------------------------------------------------
    # sklearn API
    # -------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # lock in params & remember column order
        self._sentinel = float(self.sentinel)
        self._createMissingFlags = bool(self.createMissingFlags)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df = df.fillna(self._sentinel)
        if self._createMissingFlags:
            miss_flags = df.isna().astype("int8").add_suffix("_is_missing")
            df = pd.concat([df, miss_flags], axis=1)

        return df


class TargetMomentumFeatures(BaseEstimator, TransformerMixin):
    """
    Build momentum-style features from a pre-lagged target column.

    Assumptions:
    - 'lag_col' already contains past returns.
    - This transformer never sees the true contemporaneous target.

    Features created (for each window in `windows`):
    - {prefix}_mean_{w} : rolling mean
    - {prefix}_std_{w} : rolling std
    - {prefix}_roc_{w} : rolling rate of change
    - {prefix}_cum_{w} : rolling sum

    Extra features created:
    - {prefix}_ema_fast
    - {prefix}_ema_slow
    - {prefix}_ema_diff
    - {prefix}_ema_ratio
    - {prefix}_std_21_252_ratio
    - {prefix}_lag_{k}
    
    Configurations:
    - windows: list of windows sizes
    - lags: list of lags
    
    """
    def __init__(
        self,
        lag_col:       str = "lagged_target",
        windows:       Tuple[int, ...] = (5, 10, 21, 63, 126, 252),
        lags:          Tuple[int, ...] = (1, 2, 3, 5, 10, 21, 63, 126),
        drop_source:   bool = False,
        ema_fast_span: int = 10,
        ema_slow_span: int = 50,
        min_frac:      float = 0.2,
        prefix:        str = "MOM_y",
        use_mean:      bool = True,
        use_std:       bool = True,
        use_roc:       bool = False,
        use_cum:       bool = False,
        use_ema:       bool = True,
        use_ema_ratio: bool = False,
        use_std_ratio: bool = True,
        use_lags:      bool = True,
        enabled:       bool = True,
    ) -> None:
        self.lag_col = lag_col
        self.windows = windows
        self.lags = lags
        self.drop_source = drop_source
        self.ema_fast_span = ema_fast_span
        self.ema_slow_span = ema_slow_span
        self.min_frac = min_frac
        self.prefix = prefix
        self.use_mean = use_mean
        self.use_std = use_std
        self.use_roc = use_roc
        self.use_cum = use_cum
        self.use_ema = use_ema
        self.use_ema_ratio = use_ema_ratio
        self.use_std_ratio = use_std_ratio
        self.use_lags = use_lags
        self.enabled = enabled
    

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not self.enabled:
            return self
        # lock in validated / canonicalised params
        self._windows = tuple(int(w) for w in self.windows)
        self._lags = tuple(int(l) for l in self.lags)
        self._ema_fast_span = int(self.ema_fast_span)
        self._ema_slow_span = int(self.ema_slow_span)
        self._min_frac = float(self.min_frac)
        self._prefix = str(self.prefix)

        if self.min_frac <= 0 or self.min_frac >= 1:
            raise ValueError("`min_frac` must be between 0 and 1.") 

        if self._ema_fast_span <= 0 or self._ema_slow_span <= 0:
            raise ValueError("EMA spans must be positive integers.")

        if not self._windows:
            raise ValueError("`windows` must contain at least one window.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
            
        df = X.copy()

        y_lag = self._get_base_series(df)
        feats_df = self._build_momentum_features(y_lag)

        # optionally drop the source lagged series
        if self.drop_source and self.lag_col in df.columns:
            df = df.drop(columns=[self.lag_col])

        return pd.concat([df, feats_df], axis=1)


    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _get_base_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Return cleaned, optionally log-transformed PRE-LAGGED series.
        """
        if self.lag_col not in df.columns:
            raise KeyError(f"Target column '{self.lag_col}' not found in DataFrame.")

        s = pd.to_numeric(df[self.lag_col], errors="coerce")
        return s
        # TODO: Move this to a separate DataCleaningTransformer

    def _build_momentum_features(self, y_lag: pd.Series) -> pd.DataFrame:
        feats = {}
        p = self._prefix

        for w in self._windows:
            min_periods = max(5, int(w * self._min_frac))
            roll = y_lag.rolling(w, min_periods=min_periods)

            if self.use_mean:
                feats[f"{p}_mean_{w}"] = roll.mean()
            if self.use_std:
                feats[f"{p}_std_{w}"] = roll.std()
            if self.use_roc:
                feats[f"{p}_roc_{w}"] = y_lag / y_lag.shift(w) - 1
            if self.use_cum:
                feats[f"{p}_cum_{w}"] = roll.sum()

        if self.use_ema:
            ema_fast = y_lag.ewm(span=self._ema_fast_span, adjust=False).mean()
            ema_slow = y_lag.ewm(span=self._ema_slow_span, adjust=False).mean()
            feats[f"{p}_ema_fast"] = ema_fast
            feats[f"{p}_ema_slow"] = ema_slow
            feats[f"{p}_ema_diff"] = ema_fast - ema_slow
            if self.use_ema_ratio:
                with np.errstate(divide="ignore", invalid="ignore"):
                    feats[f"{p}_ema_ratio"] = ema_fast / ema_slow - 1

        key_21 = f"{p}_std_21"
        key_252 = f"{p}_std_252"
        if self.use_std_ratio and key_21 in feats and key_252 in feats:
            with np.errstate(divide="ignore", invalid="ignore"):
                feats[f"{p}_std_21_252_ratio"] = feats[key_21] / feats[key_252]

        if self.use_lags:
            for lag in self._lags:
                feats[f"{p}_lag_{lag}"] = y_lag.shift(lag - 1)

        feats_df = pd.DataFrame(feats, index=y_lag.index).replace([np.inf, -np.inf], np.nan)
        return feats_df


class CrossSectionalFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Expand base continuous & dummy features into richer features.

    - Groups columns into categories based on prefix (Market, Price, Volatility, Sentiment, Dummy, etc.)
    - Creates features for each group

    Features created continous features (e.g. Market, Price, Volatility, Sentiment):
    - mean / std / roc / cum over rolling windows
    - EMAs and EMA-based features
    - lags
    Features created dummy features:
    - short lags
    - rolling on-rate (mean)

    Configurations:
    - windows: list of windows sizes
    - lags: list of lags
    - dummy_lags: list of dummy lags
    - dummy_windows: list of dummy windows
    - clip_roc_extremes: clip rate-of-change tails
    - roc_clip: clip rate-of-change tails
    - excluded_cols: list of columns to exclude
    - group_prefix_bucket: mapping of first-letter prefix -> bucket
    - ema_fast_span: EMA fast span
    - ema_slow_span: EMA slow span
    - min_frac: minimum fraction of non-missing values
    - use_mean: use mean
    - use_std: use standard deviation
    - use_roc: use rate of change
    - use_cum: use cumulative sum
    - use_ema: use exponential moving average
    - use_ema_ratio: use EMA ratio
    - use_lags: use lags
    - verbose: verbose output

    Notes:
    - MOM_*, VI_* and lagged_* features are *not* further extended.
    - Columns in `excluded_cols` are never touched.
    """

    def __init__(
        self,
        sentinel:            float = 0.0,
        windows:             Tuple[int, ...] = (5, 10, 21, 63, 126, 252),
        lags:                Tuple[int, ...] = (1, 2, 3, 5, 10, 21, 63, 126),
        dummy_lags:          Tuple[int, ...] = (1, 5, 21),
        dummy_windows:       Tuple[int, ...] = (5, 21),
        clip_roc_extremes:   bool = True,
        roc_clip:            float = 10.0,
        excluded_cols:       Optional[List[str]] = None,
        group_prefix_bucket: Optional[Dict[str, str]] = None,
        ema_fast_span:       int = 10,
        ema_slow_span:       int = 50,
        min_frac:            float = 0.2,
        # toggles for continuous feature explosion
        use_mean:            bool = True,
        use_std:             bool = True,
        use_roc:             bool = True,
        use_cum:             bool = False,
        use_ema:             bool = True,
        use_ema_ratio:       bool = False,
        use_lags:            bool = True,
        verbose:             bool = False,
        enabled:             bool = True,
    ):
        self.sentinel = sentinel
        self.windows = windows
        self.lags = lags
        self.dummy_lags = dummy_lags
        self.dummy_windows = dummy_windows
        self.clip_roc_extremes = clip_roc_extremes
        self.roc_clip = roc_clip
        self.excluded_cols = excluded_cols
        self.group_prefix_bucket = group_prefix_bucket
        self.ema_fast_span = ema_fast_span
        self.ema_slow_span = ema_slow_span
        self.min_frac = min_frac
        self.use_mean = use_mean
        self.use_std = use_std
        self.use_roc = use_roc
        self.use_cum = use_cum
        self.use_ema = use_ema
        self.use_ema_ratio = use_ema_ratio
        self.use_lags = use_lags
        self.verbose = verbose
        self.enabled = enabled

    # -------------------------------------------------------------
    # sklearn API
    # -------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not self.enabled:
            return self

        self._sentinel = float(self.sentinel)
        self._windows = tuple(int(w) for w in self.windows)
        self._lags = tuple(int(l) for l in self.lags)
        self._dummy_lags = tuple(int(l) for l in self.dummy_lags)
        self._dummy_windows = tuple(int(w) for w in self.dummy_windows)
        self._clip_roc_extremes = bool(self.clip_roc_extremes)
        self._roc_clip = float(self.roc_clip)
        self._excluded_cols = set(self.excluded_cols) if self.excluded_cols is not None else set()
        self._ema_fast_span = int(self.ema_fast_span)
        self._ema_slow_span = int(self.ema_slow_span)
        self._min_frac = float(self.min_frac)

        if self._ema_fast_span <= 0 or self._ema_slow_span <= 0:
            raise ValueError("EMA spans must be positive integers.")
        if not self._windows:
            raise ValueError("`windows` must contain at least one window.")

        # mapping of first-letter prefix -> bucket
        from_config = dict(self.group_prefix_bucket) if self.group_prefix_bucket is not None else dict(PREFIX_BUCKETS)
        self._prefix_buckets = from_config

        self._groups = self._create_category_groups(X)

        if self.verbose:
            counts = {k: len(v) for k, v in self._groups.items()}
            print("[FeatureBuilder] groups:", counts)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X

        df = X.copy()
        df = self._sanitize_numeric(df)

        # 1. continuous feature extension
        df = self._extend_continuous_groups(df)

        # 2. dummy feature extension
        df = self._extend_dummy_groups(df)

        # 3. clip rate-of-change tails
        if self._clip_roc_extremes:
            roc_cols = [c for c in df.columns if "_roc_" in c]
            if roc_cols:
                df.loc[:, roc_cols] = df.loc[:, roc_cols].clip(-self._roc_clip, self._roc_clip)

        return df

    # -------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------
    def _sanitize_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        feat_cols = [c for c in out.columns if c not in self._excluded_cols]
        out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan)
        return out

    def _create_category_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        groups = {
            "Market": [], "Macro": [], "Interest": [], "Price": [],
            "Volatility": [], "Sentiment": [], "Dummy": [],
            "Momentum": [], "Lagged": [], "Volatility Indicator": [],
            "Excluded": [], "Other": []
        }

        for c in df.columns:
            if c in self._excluded_cols:
                groups["Excluded"].append(c)
                continue
            if c.startswith("lagged_"):
                groups["Lagged"].append(c)
                continue
            if c.startswith("MOM"):
                groups["Momentum"].append(c)
                continue
            if c.startswith("VI"):
                groups["Volatility Indicator"].append(c)
                continue

            p = c[:1]
            bucket = self._prefix_buckets.get(p)
            if bucket is not None:
                groups[bucket].append(c)
            else:
                groups["Other"].append(c)

        return groups

    # ---------- continuous feature expansion ----------
    def _extend_continuous_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        continuous_buckets = ["Market", "Price", "Volatility", "Sentiment"]

        cont_cols: List[str] = []
        for b in continuous_buckets:
            cont_cols.extend(self._groups.get(b, []))

        cont_cols = [
            c for c in cont_cols
            if not c.startswith(("MOM", "VI", "lagged_")) and c not in self._excluded_cols
        ]

        if not cont_cols:
            return df

        frames = []
        for col in cont_cols:
            f = self._build_continuous_feature_block(df[col].astype(float), col)
            frames.append(f)

        if frames:
            extra = pd.concat(frames, axis=1)
            return pd.concat([df, extra], axis=1)
        return df

    def _build_continuous_feature_block(self, series: pd.Series, col_name: str) -> pd.DataFrame:
        s = series.shift(1)  # respect causality
        feats = {}
        p = col_name

        for w in self._windows:
            w = int(w)
            if w <= 0:
                continue
            min_periods = max(5, int(w * self._min_frac))
            roll = s.rolling(w, min_periods=min_periods)

            if self.use_mean:
                feats[f"{p}_mean_{w}"] = roll.mean()
            if self.use_std:
                feats[f"{p}_std_{w}"] = roll.std()
            if self.use_roc:
                feats[f"{p}_roc_{w}"] = s / s.shift(w) - 1
            if self.use_cum:
                feats[f"{p}_cum_{w}"] = roll.sum()

        if self.use_ema:
            ema_fast = s.ewm(span=self._ema_fast_span, adjust=False).mean()
            ema_slow = s.ewm(span=self._ema_slow_span, adjust=False).mean()
            feats[f"{p}_ema_fast"] = ema_fast
            feats[f"{p}_ema_slow"] = ema_slow
            feats[f"{p}_ema_diff"] = ema_fast - ema_slow
            if self.use_ema_ratio:
                with np.errstate(divide="ignore", invalid="ignore"):
                    feats[f"{p}_ema_ratio"] = ema_fast / ema_slow - 1

        if self.use_lags:
            for lag in self._lags:
                k = int(lag)
                if k <= 0:
                    continue
                feats[f"{p}_lag_{k}"] = s.shift(k)

        block = (
            pd.DataFrame(feats, index=series.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self._sentinel)
        )
        return block

    # ---------- dummy feature expansion ----------
    def _extend_dummy_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        d_cols = self._groups.get("Dummy", [])
        if not d_cols:
            return df

        frames = []
        for col in d_cols:
            s = df[col].astype(float).shift(1)
            ext = {}
            for k in self._dummy_lags:
                ext[f"{col}_lag_{k}"] = s.shift(k)
            for w in self._dummy_windows:
                ext[f"{col}_mean_{w}"] = s.rolling(w, min_periods=1).mean()

            fr = (
                pd.DataFrame(ext, index=df.index)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self._sentinel)
            )
            frames.append(fr)

        if frames:
            extra = pd.concat(frames, axis=1)
            return pd.concat([df, extra], axis=1)
        return df


class VolatilityIndicators(BaseEstimator, TransformerMixin):
    """
    Build volatility / regime indicators from a (pre-lagged) target series and
    optionally add regime-dependent interactions.

    Inputs:
    - lag_col: column containing PAST returns (e.g. lagged_target).
              This class never sees the true contemporaneous target.
    - Assumes (optionally) that lag_col is in simple-return space; if use_log=True,
      it is internally transformed via log1p and inverted via expm1 when needed.

    Features created:
    - VI_regime_highvol : 1 if short vol >> long vol
    - VI_regime_shock   : 1 if |return / rolling_std| > shock_z
    - VI_regime_bull    : 1 if price > SMA(bull_sma)
    - VI_volofvol_21    : pct_change of 21d vol

    Optional interactions:
    - VI_{MOM_y_ema_diff, MOM_y_roc_21, MOM_y_mean_21}_x_{bull, highvol}
    - VI_regime_highvol_x_ema

    Optional string regimes:
    - regime_vol   ∈ {low, mid, high}
    - regime_trend ∈ {bear, bull, sideways}

    Optional interaction of *raw* features with regimes:
    - {feat}_x_vol_{low,mid,high}
    - {feat}_x_trend_{bear,bull,sideways}
    """

    def __init__(
        self,
        lag_col:                 str = "lagged_target",
        sentinel:                float = 0.0,
        highvol_ratio:           float = 1.2,
        shock_z:                 float = 2.5,
        bull_sma:                int = 200,
        fallback_from_raw:       bool = True,
        add_interactions:        bool = True,
        drop_interacted_raw:     bool = True,
        drop_regime_labels:      bool = False,
        vol_interaction_feats:   Optional[List[str]] = None,
        trend_interaction_feats: Optional[List[str]] = None,
        vol_quantiles:           Tuple[float, float] = (0.25, 0.75),
        enabled:                 bool = True,
    ):
        self.lag_col = lag_col
        self.sentinel = sentinel
        self.highvol_ratio = highvol_ratio
        self.shock_z = shock_z
        self.bull_sma = bull_sma
        self.fallback_from_raw = fallback_from_raw

        self.add_interactions = add_interactions
        self.drop_interacted_raw = drop_interacted_raw
        self.drop_regime_labels = drop_regime_labels

        self.vol_interaction_feats = vol_interaction_feats
        self.trend_interaction_feats = trend_interaction_feats
        self.vol_quantiles = vol_quantiles
        self.enabled = enabled


    # -------------------------------------------------------------
    # sklearn API
    # -------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not self.enabled:
            return self
            
        self._sentinel = float(self.sentinel)
        self._highvol_ratio = float(self.highvol_ratio)
        self._shock_z = float(self.shock_z)
        self._bull_sma = int(self.bull_sma)
        self._fallback = bool(self.fallback_from_raw)
        self._lag_col = self.lag_col

        # default lists for interactions if not provided
        if self.vol_interaction_feats is None:
            self._vol_feats = ["M15", "M4", "M2", "P7", "D8", "D2", "D1"]
        else:
            self._vol_feats = list(self.vol_interaction_feats)

        if self.trend_interaction_feats is None:
            self._trend_feats = ["D6", "D7", "D8", "P12", "P8", "P7",
                                 "M17", "M4", "M2", "S5", "E12", "I4"]
        else:
            self._trend_feats = list(self.trend_interaction_feats)

        if self._bull_sma <= 0:
            raise ValueError("bull_sma must be positive.")

        if self._shock_z <= 0:
            raise ValueError("shock_z must be positive.")
            
        if self._highvol_ratio <= 0:
            raise ValueError("highvol_ratio must be positive.")

        q_low, q_high = self.vol_quantiles
        self._q_low = float(q_low)
        self._q_high = float(q_high)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X

        df = X.copy()

        # 1) base lag series
        y_lag = self._get_lagged_target(df)

        # 2) VI flags & vol-of-vol
        feat_df = self._build_vi_features(df, y_lag)

        # 3) string regimes (vol & trend)
        feat_df = self._add_string_regimes(feat_df, y_lag)

        # 4) combine base + VI features
        df_out = pd.concat([df, feat_df], axis=1)

        # 5) optional regime interactions with selected raw features
        if self.add_interactions:
            df_out = self._add_regime_interactions(df_out)

        # 6) optionally drop regime label columns
        if self.drop_regime_labels:
            drop_cols = [c for c in ["regime_vol", "regime_trend"] if c in df_out.columns]
            if drop_cols:
                df_out = df_out.drop(columns=drop_cols)

        return df_out

    # -------------------------------------------------------------
    # internals
    # -------------------------------------------------------------
    def _get_lagged_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Return cleaned, optionally log-transformed PRE-LAGGED series.
        """
        if self.lag_col not in df.columns:
            raise KeyError(f"Target column '{self.lag_col}' not found in DataFrame.")

        lagged_target = pd.to_numeric(df[self.lag_col], errors="coerce")
        return lagged_target

    def _build_vi_features(self, df: pd.DataFrame, y_lag: Optional[pd.Series]) -> pd.DataFrame:
        feat: Dict[str, pd.Series] = {}

        # ----- high vol flag from ratio of short/long vol -----
        std21_252_ratio = df.get("MOM_y_std_21_252_ratio")
        if std21_252_ratio is not None:
            feat["VI_regime_highvol"] = (
                std21_252_ratio.fillna(-np.inf) > self._highvol_ratio
            ).astype("int8")
        else:
            std21 = df.get("MOM_y_std_21")
            std252 = df.get("MOM_y_std_252")
            if std21 is not None and std252 is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = std21 / std252
                feat["VI_regime_highvol"] = (ratio > self._highvol_ratio).astype("int8")
            else:
                feat["VI_regime_highvol"] = pd.Series(0, index=df.index, dtype="int8")

        # ----- shock flag -----
        if y_lag is not None and not y_lag.empty:
            den = y_lag.rolling(21, min_periods=5).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                z = y_lag / den
            feat["VI_regime_shock"] = (
                z.abs().fillna(-np.inf) > self._shock_z
            ).astype("int8")
        else:
            feat["VI_regime_shock"] = pd.Series(0, index=df.index, dtype="int8")

        # ----- bull regime flag & price series -----
        if y_lag is not None and not y_lag.empty:
            # convert back to simple returns if log
            r = y_lag.fillna(0)
            mkt = (1.0 + r).cumprod()
            sma200 = mkt.rolling(self._bull_sma, min_periods=20).mean()
            feat["VI_regime_bull"] = (
                mkt.fillna(-np.inf) > sma200.fillna(-np.inf)
            ).astype("int8")
        else:
            # still define, but always zero
            mkt = pd.Series(1.0, index=df.index)
            feat["VI_regime_bull"] = pd.Series(0, index=df.index, dtype="int8")

        # ----- vol-of-vol -----
        std21 = df.get("MOM_y_std_21")
        if std21 is not None:
            feat["VI_volofvol_21"] = (
                std21.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            )
        else:
            feat["VI_volofvol_21"] = pd.Series(0.0, index=df.index)

        # ----- momentum-based interactions -----
        for col in ["MOM_y_ema_diff", "MOM_y_roc_21", "MOM_y_mean_21"]:
            if col in df:
                feat[f"VI_{col}_x_bull"] = df[col] * feat["VI_regime_bull"]
                feat[f"VI_{col}_x_highvol"] = df[col] * feat["VI_regime_highvol"]

        ema_diff = df.get("MOM_y_ema_diff")
        if ema_diff is not None:
            feat["VI_regime_highvol_x_ema"] = feat["VI_regime_highvol"] * ema_diff
        else:
            feat["VI_regime_highvol_x_ema"] = pd.Series(0.0, index=df.index)

        feat_df = (
            pd.DataFrame(feat, index=df.index)
            .replace([np.inf, -np.inf], np.nan)
        )

        # stash price in case we need it for trend regimes
        feat_df["_VI_price"] = mkt
        return feat_df

    def _add_string_regimes(self, feat_df: pd.DataFrame, y_lag: Optional[pd.Series]) -> pd.DataFrame:
        # default: neutral labels if we have no usable series
        if y_lag is None or y_lag.empty:
            feat_df["regime_vol"] = "mid"
            feat_df["regime_trend"] = "sideways"
            return feat_df

        # volatility regimes based on realized vol of y_lag
        roll_std_21 = y_lag.rolling(21, min_periods=15).std()
        roll_valid = roll_std_21.dropna()
        if len(roll_valid) == 0:
            feat_df["regime_vol"] = "mid"
        else:
            hi_thresh = roll_valid.quantile(self._q_high)
            lo_thresh = roll_valid.quantile(self._q_low)

            feat_df["regime_vol"] = "mid"
            high_idx = roll_valid[roll_valid >= hi_thresh].index
            low_idx = roll_valid[roll_valid <= lo_thresh].index
            feat_df.loc[high_idx, "regime_vol"] = "high"
            feat_df.loc[low_idx, "regime_vol"] = "low"

        # trend regimes based on stored price
        price = feat_df.get("_VI_price")
        if price is None:
            feat_df["regime_trend"] = "sideways"
        else:
            sma_200 = price.rolling(200, min_periods=50).mean()
            trend_valid = price.notna() & sma_200.notna()
            price_valid = price[trend_valid]
            sma_valid = sma_200[trend_valid]

            feat_df["regime_trend"] = "sideways"
            bull_idx = price_valid[price_valid > sma_valid * 1.01].index
            bear_idx = price_valid[price_valid < sma_valid * 0.99].index
            feat_df.loc[bull_idx, "regime_trend"] = "bull"
            feat_df.loc[bear_idx, "regime_trend"] = "bear"

        # drop internal price helper column
        if "_VI_price" in feat_df.columns:
            feat_df = feat_df.drop(columns=["_VI_price"])

        return feat_df

    def _add_regime_interactions(self, df_out: pd.DataFrame) -> pd.DataFrame:
        to_drop = set()
        base = df_out  # use full df_out for all interactions

        # vol x feature
        if "regime_vol" in df_out.columns:
            vol_regs = ["low", "mid", "high"]
            for f in self._vol_feats:
                if f not in base.columns:
                    continue
                for reg in vol_regs:
                    mask = (df_out["regime_vol"] == reg).astype("int8")
                    df_out[f"{f}_x_vol_{reg}"] = base[f] * mask
                if self.drop_interacted_raw and f in base.columns:
                    to_drop.add(f)

        # trend x feature
        if "regime_trend" in df_out.columns:
            trend_regs = ["bear", "bull", "sideways"]
            for f in self._trend_feats:
                if f not in base.columns:
                    continue
                for reg in trend_regs:
                    mask = (df_out["regime_trend"] == reg).astype("int8")
                    df_out[f"{f}_x_trend_{reg}"] = base[f] * mask
                if self.drop_interacted_raw and f in base.columns:
                    to_drop.add(f)

        if self.drop_interacted_raw and to_drop:
            df_out = df_out.drop(columns=list(to_drop))

        return df_out

class dropExcludedCols(BaseEstimator, TransformerMixin):
    def __init__(self,
                 cols_to_drop: list[str]):
        self.cols_to_drop = cols_to_drop
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(self.cols_to_drop, axis=1, errors="ignore")

class PrecomputedTopKSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_ranking: list[str], k: int = 50, verbose: bool = False):
        """
        feature_ranking: global ranking of features (highest importance / corr first)
        k: number of top features to keep
        """
        self.feature_ranking = feature_ranking
        self.k = k
        self.verbose = verbose

    def fit(self, X, y=None):
        # Restrict ranking to features that actually exist in X
        self.feature_ranking_in_X_ = [
            f for f in self.feature_ranking if f in X.columns
        ]
        return self

    def transform(self, X):
        # Safety: if someone skipped fit, recover gracefully
        if not hasattr(self, "feature_ranking_in_X_"):
            self.fit(X)

        # Intersect again in case columns changed between fit and transform
        ranking_in_X = [f for f in self.feature_ranking_in_X_ if f in X.columns]

        if self.k is None:
            selected = ranking_in_X
        else:
            selected = ranking_in_X[: self.k]

        if self.verbose:
            print(f"[Selector] k={self.k} | Input cols: {X.shape[1]} | Selected: {len(selected)}")
        
        if not selected:
             raise ValueError("No selected features found in X.columns!")

        return X[selected]
        if not selected:
            raise ValueError(
                "[PrecomputedTopKSelector] After intersecting with X.columns, "
                "no features remain to select."
            )

        return X[selected]

    def get_params(self, deep=True):
        # Only expose k as a hyperparameter; feature_ranking is fixed metadata
        return {"k": self.k, "feature_ranking": self.feature_ranking}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class PCATransformer(BaseEstimator, TransformerMixin):
    """
    Apply PCA to a subset (or all) of columns, return a DataFrame.

    Parameters
    ----------
    columns : list[str] or None
        Columns to apply PCA on. If None, use all columns.
    n_components : int or float
        If int: number of components.
        If float in (0, 1): fraction of variance for PCA to keep.
    keep_other : bool
        If True, keep non-PCA columns and append PCA components.
        If False, output only PCA components.
    prefix : str
        Prefix for PCA component column names.
    standardize : bool
        If True, standardizes selected columns before PCA.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        n_components: int | float = 20,
        keep_other: bool = False,
        prefix: str = "PCA",
        standardize: bool = True,
    ):
        self.columns = columns
        self.n_components = n_components
        self.keep_other = keep_other
        self.prefix = prefix
        self.standardize = standardize

    def fit(self, X, y=None):
        df = self._to_df(X)

        # Decide which columns to apply PCA to
        if self.columns is None:
            cols = list(df.columns)
        else:
            cols = [c for c in self.columns if c in df.columns]

        if not cols:
            raise ValueError("PCATransformer: no valid columns to apply PCA on.")

        self._cols_ = cols

        X_block = df[cols].values

        # Standardize if requested
        if self.standardize:
            self._scaler_ = StandardScaler()
            X_block = self._scaler_.fit_transform(X_block)
        else:
            self._scaler_ = None

        # Configure PCA
        n_comp = self.n_components
        if isinstance(n_comp, float) and 0 < n_comp < 1:
            # variance fraction
            self._pca_ = PCA(n_components=n_comp, svd_solver="full")
        else:
            n_comp = min(int(n_comp), X_block.shape[1])
            self._pca_ = PCA(n_components=n_comp)

        self._pca_.fit(X_block)

        # Keep column names for components
        self._component_names_ = [
            f"{self.prefix}_{i+1}" for i in range(self._pca_.n_components_)
        ]

        return self

    def transform(self, X):
        df = self._to_df(X)
        X_block = df[self._cols_].values

        if getattr(self, "_scaler_", None) is not None:
            X_block = self._scaler_.transform(X_block)

        comps = self._pca_.transform(X_block)
        comp_df = pd.DataFrame(
            comps,
            index=df.index,
            columns=self._component_names_,
        )

        if self.keep_other:
            rest = df.drop(columns=self._cols_, errors="ignore")
            out = pd.concat([rest, comp_df], axis=1)
        else:
            out = comp_df

        return out

    @staticmethod
    def _to_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

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


def SortFeaturesByCorrElastic(X_train, y_train):
    # 1) build + fit full pipeline
    pipe = create_pipeline(
        make_elasticnet(alpha=0.0009609238331494418,
                        l1_ratio=0.5419274494337433)
    )
    pipe.set_params(**FEATURE_LEVEL_CONFIGS["extensive"])
    pipe.fit(X_train, y_train)

    # 2) get preprocessor (all but final model)
    preproc = pipe[:-1]
    X_trans = preproc.transform(X_train)

    # 3) feature names after preprocessing
    if isinstance(X_trans, pd.DataFrame):
        feature_names = np.array(X_trans.columns)
    else:
        feature_names = np.array([f"feat_{i}" for i in range(X_trans.shape[1])])

    # 4) get the actual ElasticNet estimator
    final_est = get_final_estimator(pipe.named_steps["model"])
    coefs = final_est.coef_

    # 5) rank by |coef|
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

    # preprocessor
    preproc = pipe[:-1]
    X_trans = preproc.transform(X_train)

    if isinstance(X_trans, pd.DataFrame):
        feature_names = np.array(X_trans.columns)
    else:
        feature_names = np.array([f"feat_{i}" for i in range(X_trans.shape[1])])

    # handle possible nested pipeline around XGB
    final_est = get_final_estimator(pipe.named_steps["model"])
    importances = final_est.feature_importances_

    idx_sorted = np.argsort(importances)[::-1]
    sorted_features = feature_names[idx_sorted]

    return list(sorted_features)

def ts_cross_val_score(pipe, X, y, n_splits=3, use_early_stopping=False, es_rounds=50):
    """
    Plain time-series CV using sklearn's cross_val_score.
    Early stopping is disabled because the XGBoost sklearn wrapper
    in this environment does not support early_stopping_rounds/callbacks.
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


    # 1) choose feature level
    # if model_type in ("ols", "ridge", "lasso", "elastic"):
    #     feat_levels = ["none", "simple", "simple2", "medium"]
    # elif model_type in ("lgbm", "xgb"):
    #     feat_levels = ["medium", "advanced", "extensive"]
    # else:
    #     feat_levels = ["none", "simple", "simple2", "medium", "advanced", "extensive"]

    # choose feature level
    # if not fixed_features:
    #     feature_level = trial.suggest_categorical("feature_level", feat_levels)
    # else:
    #     idx = trial.number % len(feat_levels)
    #     feature_level = feat_levels[idx]
    #     # log it so it shows up in trials_dataframe()
    
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

    #     # Cross-sectional feature tweaks
    #     pipe.set_params(
    #         features__use_roc=trial.suggest_categorical(
    #             "features__use_roc", [True, False]
    #         ),
    #         features__use_cum=trial.suggest_categorical(
    #             "features__use_cum", [False, True]
    #         ),
    #     )

    #     # Volatility module toggles (if you want)
    #     pipe.set_params(
    #         vol__add_interactions=trial.suggest_categorical(
    #             "vol__add_interactions", [True, False]
    #         )
    #     )

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


# study_xgb = run_optuna_for_model(
#     X=X_cond,
#     y=y_cond,
#     model_type="xgb",
#     n_trials=80,
#     n_splits=3,
#     direction="maximize",  # if using R² or Sharpe-like
# )

# study_ridge = run_optuna_for_model(
#     X=X_cond,
#     y=y_cond,
#     model_type="ridge",
#     n_trials=50,
#     n_splits=3,
# )


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

def time_train_test_split(X, y, test_frac: float = 0.2):
    """
    Chronological split: first (1 - test_frac) for train, tail for test.
    """
    n = len(X)
    split = int(n * (1 - test_frac))
    X_train = X.iloc[:split].copy()
    X_test  = X.iloc[split:].copy()
    y_train = y.iloc[:split].copy()
    y_test  = y.iloc[split:].copy()
    return X_train, X_test, y_train, y_test

# ----------------------------------------------------------------------------------------
# CALL STUDIES
# ----------------------------------------------------------------------------------------

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
    n_trials=300,
    n_splits=3,
    prune_features=False,
)

out_dir = "optuna_results"
save_studies_to_disk(studies, "final")


# # ==============================================================================
# # 2. PCA & HYBRID EXPERIMENTS
# # ==============================================================================

# def infer_pca_block_columns(X_sample: pd.DataFrame) -> list[str]:
#     """
#     Select columns for 'PCA + raw' hybrid approach.
#     Excluded: EXCLUDED_COLS, regimes, lagged targets, dummies.
#     """
#     cols = []
#     for c in X_sample.columns:
#         if c in EXCLUDED_COLS:
#             continue
#         if c.startswith(("VI_regime_", "regime_", "lagged_", "D")):
#             continue
#         cols.append(c)
#     return cols

# def columns_with_prefix(X_sample: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
#     """
#     Select columns matching specific prefixes (e.g. ('M',) for Market).
#     """
#     cols = []
#     for c in X_sample.columns:
#         if any(c.startswith(p) for p in prefixes):
#             cols.append(c)
#     return cols

# def run_pca_experiments(X: pd.DataFrame, y: pd.Series):
#     print("\n========================================")
#     print("Running PCA Experiments")
#     print("========================================")
    
#     # 1. Split
#     X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_frac=0.2)
    
#     # Helper to get feature names / columns from a 'medium' run
#     print("...running feature generation on train set to identify columns...")
#     tmp_pipe = create_pipeline(make_ridge())
#     tmp_pipe.set_params(**FEATURE_LEVEL_CONFIGS["medium"])
    
#     # Fit preprocessor only (slice up to 'model')
#     # steps: base_cleaner, target_lag, momentum, features, vol, drop, imputer, model
#     # We want everything EXCEPT model
#     preproc_steps = tmp_pipe.steps[:-1]
#     preproc_pipe = Pipeline(preproc_steps)
    
#     preproc_pipe.fit(X_train, y_train)
#     X_train_feat = preproc_pipe.transform(X_train)
    
#     if not isinstance(X_train_feat, pd.DataFrame):
#         X_train_feat = pd.DataFrame(X_train_feat, index=X_train.index)

#     # ---------------------------------------------------------
#     # Option 1: PCA-only regression
#     # ---------------------------------------------------------
#     print("\n[PCA Option 1] PCA-only (Ridge)")
#     ridge_model1 = make_ridge(alpha=0.1)
    
#     pca_only = PCATransformer(
#         columns=None,          # all columns
#         n_components=20,       
#         keep_other=False,      
#         prefix="PC",
#         standardize=True,
#     )
    
#     pipe_pca_only = create_pca_pipeline(
#         model=ridge_model1,
#         selector=None, 
#         pca_transformer=pca_only,
#     )
#     pipe_pca_only.set_params(**FEATURE_LEVEL_CONFIGS["medium"])
    
#     pipe_pca_only.fit(X_train, y_train)
#     y_pred1 = pipe_pca_only.predict(X_test)
#     r2_1 = r2_score(y_test, y_pred1)
#     print(f"R2 PCA-only: {r2_1:.5f}")

#     # ---------------------------------------------------------
#     # Option 2: PCA + raw features
#     # ---------------------------------------------------------
#     print("\n[PCA Option 2] Hybrid PCA + Raw (Ridge)")
    
#     pca_cols_opt2 = infer_pca_block_columns(X_train_feat)
#     print(f"Hybrid PCA: compressing {len(pca_cols_opt2)} columns, keeping others raw.")

#     ridge_model2 = make_ridge(alpha=0.1)
    
#     pca_hybrid = PCATransformer(
#         columns=pca_cols_opt2,
#         n_components=20,
#         keep_other=True,
#         prefix="PC_blk",
#         standardize=True,
#     )
    
#     pipe_pca_hybrid = create_pca_pipeline(
#         model=ridge_model2,
#         selector=None,
#         pca_transformer=pca_hybrid,
#     )
#     pipe_pca_hybrid.set_params(**FEATURE_LEVEL_CONFIGS["medium"]) 
    
#     pipe_pca_hybrid.fit(X_train, y_train)
#     y_pred2 = pipe_pca_hybrid.predict(X_test)
#     r2_2 = r2_score(y_test, y_pred2)
#     print(f"R2 PCA-hybrid: {r2_2:.5f}")

#     # ---------------------------------------------------------
#     # Option 3: Block / Group PCA
#     # ---------------------------------------------------------
#     print("\n[PCA Option 3] Block PCA (Ridge)")
    
#     market_cols = columns_with_prefix(X_train_feat, ("M",))
#     price_cols  = columns_with_prefix(X_train_feat, ("P",))
#     vol_cols    = columns_with_prefix(X_train_feat, ("V",))
#     sent_cols   = columns_with_prefix(X_train_feat, ("S",))
    
#     print(f"Blocks: M={len(market_cols)}, P={len(price_cols)}, V={len(vol_cols)}, S={len(sent_cols)}")
    
#     ridge_model3 = make_ridge(alpha=0.1)
    
#     # Manually constructing pipeline to allow multiple PCAs
#     steps_base = [
#         ("base_cleaner", BaseCleaner()),
#         ("target_lag", TargetLagBuilder(target_col=TARGET_COL, lag_col="lagged_target", drop_target=True)),
#         ("momentum", TargetMomentumFeatures()),
#         ("features", CrossSectionalFeatureBuilder(sentinel=0.0)),
#         ("vol", VolatilityIndicators()),
#         ("drop", dropExcludedCols(cols_to_drop=list(EXCLUDED_COLS))),
#         ("imputer", MissingValueImputer(sentinel=0.0)),
#     ]
    
#     steps_pca_blocks = steps_base + [
#         ("pca_market", PCATransformer(columns=market_cols, n_components=5, keep_other=True, prefix="PC_M")),
#         ("pca_price",  PCATransformer(columns=price_cols,  n_components=5, keep_other=True, prefix="PC_P")),
#         ("pca_vol",    PCATransformer(columns=vol_cols,    n_components=3, keep_other=True, prefix="PC_V")),
#         ("pca_sent",   PCATransformer(columns=sent_cols,   n_components=3, keep_other=True, prefix="PC_S")),
#         ("model", ridge_model3),
#     ]
    
#     pipe_block_pca = Pipeline(steps_pca_blocks)
#     pipe_block_pca.set_params(**FEATURE_LEVEL_CONFIGS["medium"])
    
#     pipe_block_pca.fit(X_train, y_train)
#     y_pred3 = pipe_block_pca.predict(X_test)
#     r2_3 = r2_score(y_test, y_pred3)
#     print(f"R2 PCA-Block: {r2_3:.5f}")
    
# run_pca_experiments(X, y)
