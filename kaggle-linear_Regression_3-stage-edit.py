import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Tuple

# ----------------------------------------------------
# This program is divided into the following six section
# 1) Load Data
# 2) Missing value processing
# 3) Feature Engineering
#  Add features that represent historical information.
# 4) Feature selection (stage 1)
#  Run Lasso regression with small alpha for each feature group and keep only the features that were selected
# 5) Feature selection (stage 2)
#  Apply Lasso with a small alpha to the features selected in Stage 1. Then, examine the coefficients for each fold and keep only the features that are stable across folds
# 6) Final regression (stage 3 : evaluation)
#  Evaluate the features selected in Stage 2 using Elastic Net, Ridge, and Lasso.
# ----------------------------------------------------

# 0) Display sanity
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 1000)

# 1) Load Data

df = pd.read_csv('kaggle/input/Hull-Tactical-Market-Prediction/train.csv')
df = df.sort_values("date_id")


# 2) Missing value processing

# drop first 1006 rows with many missing values
df = df[1006:] # df[1006:] or df[1511:]

# fullfil missing value with mean of other features in the same row
target_cols = ['date_id','market_forward_excess_returns', 'risk_free_rate', 'forward_returns']
feature_cols = [c for c in df.columns if c not in target_cols]
row_means = df[feature_cols].mean(axis=1, skipna=True)

for col in feature_cols:
    mask = df[col].isna()
    if mask.any():
        # missing flag
        df[col + "_missing"] = mask.astype(int)
        # fullfil missing value
        df.loc[mask, col] = row_means[mask]



# 3) Feature Engineering
# Add features that represent historical information.

df["excess_retuen_lag1"]  = df["market_forward_excess_returns"].shift(1)  # add lag return to features

df=df.drop(['D1','D2','D3','D4','D5','D6','D7','D8','D9'],axis=1) # drop all the dummy / binary features 
df=df.drop(["M4"],axis=1) # dropping this feature improves Test R^2


# Separate missing-flag columns and actual feature columns 
features_with_missing_flag = df.drop(['date_id','market_forward_excess_returns', 'risk_free_rate', 'forward_returns'], axis=1)

missing_flag_col = [c for c in features_with_missing_flag.columns if c.endswith("_missing")]
features_col = [c for c in features_with_missing_flag.columns if not c.endswith("_missing")]

missing_flag = features_with_missing_flag[missing_flag_col]
features   = features_with_missing_flag[features_col]

# -----------create features representing historical information-----------

df_lag1 = features.shift(1).add_suffix("_lag1") 
df_ma5 = features.rolling(window=5).mean().add_suffix("_ma5")
df_vol5 = features.rolling(5).std().add_suffix("_vol5")
df_vol5 = features.ewm(span=5, adjust=False).std().add_suffix("_vol5")
df_abs = features.abs().add_suffix("_abs")
df_abs_ma5 = features.abs().rolling(window=5).mean().add_suffix("_abs_ma5")


# Difference compared to yesterday
df_diff1 = features.diff(1).add_suffix("_diff1")
df_diff1_abs = features.diff(1).abs().add_suffix("_diff1_abs")


# Difference compared to 21 working days ago (= approximately 1 month)
df_diff21 = features.diff(21).ewm(span=5, adjust=False).mean().add_suffix("_diff21") # using an exponentially weighted mean for smoothing
df_diff21_abs = features.diff(21).ewm(span=5, adjust=False).mean().abs().add_suffix("_diff21_abs")

# Difference compared to 63 working days ago (= approximately 3 month)
df_diff63 = features.diff(63).ewm(span=10, adjust=False).mean().add_suffix("_diff63")
df_diff63_abs = features.diff(63).ewm(span=10, adjust=False).mean().abs().add_suffix("_diff63_abs")

df["V13_diff1_diff1_abs"] = df["V13"].diff(1).diff(1).abs() # this feature improves both train and test R^2, but I'm not sure why it works
features_with_missing_flag["V13_diff1_diff1_abs"]=df["V13_diff1_diff1_abs"] 

df = pd.concat([df, df_ma5 , df_diff1 , df_diff1_abs,df_diff21,df_diff21_abs, df_diff63,df_diff63_abs], axis=1) # Add the generated features to the dataset.
df = pd.concat([df, df_abs,df_abs_ma5,df_lag1,df_vol5], axis=1) # Add the generated features to the dataset.

df = df.dropna().reset_index(drop=True) # drop NaN value created when generating historical features 



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
        drop_interacted_raw:     bool = False,
        drop_regime_labels:      bool = True,
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

        # # ----- high vol flag from ratio of short/long vol -----
        # std21_252_ratio = df.get("MOM_y_std_21_252_ratio")
        # if std21_252_ratio is not None:
        #     feat["VI_regime_highvol"] = (
        #         std21_252_ratio.fillna(-np.inf) > self._highvol_ratio
        #     ).astype("int8")
        # else:
        #     std21 = df.get("MOM_y_std_21")
        #     std252 = df.get("MOM_y_std_252")
        #     if std21 is not None and std252 is not None:
        #         with np.errstate(divide="ignore", invalid="ignore"):
        #             ratio = std21 / std252
        #         feat["VI_regime_highvol"] = (ratio > self._highvol_ratio).astype("int8")
        #     else:
        #         feat["VI_regime_highvol"] = pd.Series(0, index=df.index, dtype="int8")

        # # ----- shock flag -----
        # if y_lag is not None and not y_lag.empty:
        #     den = y_lag.rolling(21, min_periods=5).std()
        #     with np.errstate(divide="ignore", invalid="ignore"):
        #         z = y_lag / den
        #     feat["VI_regime_shock"] = (
        #         z.abs().fillna(-np.inf) > self._shock_z
        #     ).astype("int8")
        # else:
        #     feat["VI_regime_shock"] = pd.Series(0, index=df.index, dtype="int8")

        # # ----- bull regime flag & price series -----
        # if y_lag is not None and not y_lag.empty:
        #     # convert back to simple returns if log
        #     r = y_lag.fillna(0)
        #     mkt = (1.0 + r).cumprod()
        #     sma200 = mkt.rolling(self._bull_sma, min_periods=20).mean()
        #     feat["VI_regime_bull"] = (
        #         mkt.fillna(-np.inf) > sma200.fillna(-np.inf)
        #     ).astype("int8")
        # else:
        #     # still define, but always zero
        #     mkt = pd.Series(1.0, index=df.index)
        #     feat["VI_regime_bull"] = pd.Series(0, index=df.index, dtype="int8")

        # # ----- vol-of-vol -----
        # std21 = df.get("MOM_y_std_21")
        # if std21 is not None:
        #     feat["VI_volofvol_21"] = (
        #         std21.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        #     )
        # else:
        #     feat["VI_volofvol_21"] = pd.Series(0.0, index=df.index)

        # # ----- momentum-based interactions -----
        # for col in ["MOM_y_ema_diff", "MOM_y_roc_21", "MOM_y_mean_21"]:
        #     if col in df:
        #         feat[f"VI_{col}_x_bull"] = df[col] * feat["VI_regime_bull"]
        #         feat[f"VI_{col}_x_highvol"] = df[col] * feat["VI_regime_highvol"]

        # ema_diff = df.get("MOM_y_ema_diff")
        # if ema_diff is not None:
        #     feat["VI_regime_highvol_x_ema"] = feat["VI_regime_highvol"] * ema_diff
        # else:
        #     feat["VI_regime_highvol_x_ema"] = pd.Series(0.0, index=df.index)

        feat_df = (
            pd.DataFrame(feat, index=df.index)
            .replace([np.inf, -np.inf], np.nan)
        )

        # # stash price in case we need it for trend regimes
        # feat_df["_VI_price"] = mkt
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

y = df['market_forward_excess_returns'] # target
X = df.drop(['date_id','market_forward_excess_returns', 'risk_free_rate', 'forward_returns'], axis=1).astype('float64') # features

vi = VolatilityIndicators(lag_col='excess_retuen_lag1')
X = vi.fit_transform(X)

print(X.columns)

# 4) Feature selection (stage 1)
# Run Lasso regression with small alpha for each feature group and keep only the features that were selected



# Split the data into train and test sets
n = len(X)
split_size = int(n * 0.8) 

X_train = X[:split_size]
y_train = y[:split_size]
X_test  = X[split_size:]
y_test  = y[split_size:]

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)



# Define a function that returns the features selected by Lasso
def get_lasso_selected_features(
    X: pd.DataFrame, 
    y: pd.Series,
    alphas=np.logspace(-6, -2, 30),
    cv=TimeSeriesSplit(n_splits=3),
    random_state=0
):


    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(
            alphas=alphas,
            cv=cv, random_state=random_state,max_iter=100000))
    ])
    model.fit(X, y)
    lasso = model.named_steps["lasso"]

    coef = lasso.coef_
    feature_names = X.columns

    selected = [(name, c) for name, c in zip(feature_names, coef) if abs(c) > 1e-15]

    df_selected = pd.DataFrame(selected, columns=["feature", "coef"])
    df_selected = df_selected.reindex(df_selected['coef'].abs().sort_values(ascending=False).index)

    # print selected features
    print("===== Lasso Selected Features =====")
    if len(df_selected) == 0:
        print("No features selected.")
    else:
        print(df_selected.to_string(index=False))
    print("Optimal alpha:", lasso.alpha_)
    print("===================================\n")    
 
    result = {
        "selected_df": df_selected,
        "selected_features": df_selected["feature"].tolist(),
        "n_selected": len(df_selected),
        "alpha": lasso.alpha_,
        "coefficients": coef,
        "intercept": lasso.intercept_,
        "model": model
    }
    return result

# Run feature selection
res_df= get_lasso_selected_features(X_train,y_train,alphas=np.logspace(-5,-3.5,20))
res_features= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)],y_train,alphas=np.logspace(-5,-3.5,20))
res_lag1= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_lag1.columns)],y_train,alphas=np.logspace(-5,-3.5,20))
res_ma5= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_ma5.columns)],y_train,alphas=np.logspace(-5,-3.5,20))
res_vol5= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_vol5.columns)],y_train,alphas=np.logspace(-5,-3.5,20))
res_abs= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_abs.columns)],y_train,alphas=np.logspace(-5,-3.5,20))
res_abs_ma5= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_abs_ma5.columns)],y_train,alphas=np.logspace(-5,-3.5,20))
res_diff1= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_diff1.columns)],y_train,alphas=np.logspace(-5,-3.5,20))
res_diff1_abs= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_diff1_abs.columns)],y_train,alphas=np.logspace(-5,-3.5,20))
res_diff21= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_diff21.columns)],y_train,alphas=np.logspace(-5,-3.2,20))
res_diff21_abs= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_diff21_abs.columns)],y_train,alphas=np.logspace(-5,-3.2,20))
res_diff63= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_diff63.columns)],y_train,alphas=np.logspace(-5,-3,20))
res_diff63_abs= get_lasso_selected_features(X_train[list(features_with_missing_flag.columns)+list(df_diff63_abs.columns)],y_train,alphas=np.logspace(-5,-3,20))

# Combine the results
selected_features=list(dict.fromkeys(
#    list(features_with_missing_flag.columns)
    list(res_features["selected_features"])
#    +list(res_lag1["selected_features"])
    +list(res_ma5["selected_features"])
#    +list(res_vol5["selected_features"])
#    +list(res_abs["selected_features"])
#    +list(res_abs_ma5["selected_features"])
    +list(res_diff1["selected_features"])
    +list(res_diff1_abs["selected_features"])
#    +list(res_diff21["selected_features"])
#    +list(res_diff21_abs["selected_features"])
#    +list(res_diff63["selected_features"])
#    +list(res_diff63_abs["selected_features"])
))

print("selected features: ",selected_features)

# Replace the features with the selected features
X_train=X_train[selected_features]
X_test=X_test[selected_features]


# 5) Feature selection (stage 2)
# Apply Lasso with a small alpha to the features selected in Stage 1. Then, examine the coefficients for each fold and keep only the features that are stable across folds

tscv = TimeSeriesSplit(n_splits=5)  

# ------------ Lasso Regression ------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(
        alphas=np.logspace(-5,-4,30), # use small alpha
        cv=tscv, random_state=0,max_iter=100000))
])
model.fit(X_train, y_train)

lasso = model.named_steps["lasso"]
mse_path = lasso.mse_path_  

y_var = np.var(y_train, ddof=1)
cv_r2 = 1 - mse_path / y_var  

alpha_idx = list(lasso.alphas_).index(lasso.alpha_)

print("R² for each CV split at the best alpha:")
for i, r2 in enumerate(cv_r2[alpha_idx]):
    print(f"  CV{i+1}: {r2:.6f}")

best_alpha = model.named_steps["lasso"].alpha_
print("Best alpha (LassoCV):", best_alpha)

y_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_pred)

print("Train R^2 (LassoCV):", r2_train)

y_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_pred)

print("Test R^2(LassoCV):", r2_test)
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": model.named_steps["lasso"].coef_
})

threshold = 1e-20
nonzero_df = coef_df[coef_df["coef"].abs() > threshold]
print(nonzero_df.sort_values("coef", ascending=False))


# Retrieve the Lasso coefficients for each fold
from sklearn.linear_model import Lasso
def lasso_coef_by_fold_pipeline(X, y, alpha, n_splits=5):

    cv = TimeSeriesSplit(n_splits=5)

    coef_list = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X), start=1):

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=alpha, max_iter=10000))
        ])

        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        lasso = pipe.named_steps["lasso"]
        coef = lasso.coef_

        s = pd.Series(coef, index=X.columns, name=f"fold{fold_idx}")
        coef_list.append(s)

    coef_df = pd.concat(coef_list, axis=1).T
    return coef_df

coef_df = lasso_coef_by_fold_pipeline(X_train, y_train, alpha=best_alpha, n_splits=5)

print(coef_df.T)   

coef_mean = coef_df.mean(axis=0)
coef_std  = coef_df.std(axis=0)

coef_summary = pd.DataFrame({
    "mean": coef_mean,
    "std": coef_std
})

# Sort by absolute mean value
print(
    coef_summary.loc[coef_summary["mean"].abs() > 1e-8]
                 .sort_values("mean", key=lambda x: x.abs(), ascending=False)
)

coef_mean = coef_df.mean(axis=0)
coef_std  = coef_df.std(axis=0)

# Stable features : abs(mean) > std * 0.5
stable_features = coef_mean.index[coef_mean.abs() > coef_std *0.5]

# Unstable features：abs(mean) <= std * 0.5
unstable_features = coef_mean.index[coef_mean.abs() <= coef_std *0.5 ]

print("Stable features (selected):")
print(list(stable_features))

print("\nUnstable features (dropped):")
print(list(unstable_features))


# Replace the features with the selected features
X_train = X_train[list(stable_features)]
X_test  = X_test[list(stable_features)]

print("X_train.shape",X_train.shape)
print("X_test.shape",X_test.shape)

# 6) Final regression (evaluation)
# Evaluate the features selected in Stage 2 using Elastic Net, Ridge, and Lasso.
  
tscv = TimeSeriesSplit(n_splits=5)  

# ------------ Elastic Net ------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNetCV(
        l1_ratio=np.linspace(0.01,0.30,20),  
        alphas=np.logspace(-3,-0.5,20),  
        cv=tscv,
        random_state=0,
        max_iter=100000
    ))
])

model.fit(X_train, y_train)
enet = model.named_steps["enet"]

# Compute R² for each CV split 
mse_path = enet.mse_path_  # shape = (n_l1_ratio, n_alpha, n_cv)
y_var = np.var(y_train, ddof=1)

cv_r2 = 1 - mse_path / y_var  # Convert MSE to R²
cv_r2_mean = cv_r2.mean(axis=0)  # Average over l1_ratio → shape = (n_alpha, n_cv)

# Get the index of the best alpha
alpha_idx = list(enet.alphas_).index(enet.alpha_)

# Display R² for each CV split at the best alpha
print("R² for each CV split at the best alpha:")
for i, r2 in enumerate(cv_r2_mean[alpha_idx]):
    print(f"  CV{i+1}: {r2:.6f}")


best_alpha = model.named_steps["enet"].alpha_
best_l1_ratio = model.named_steps["enet"].l1_ratio_
print("Best alpha (ElasticNet):", best_alpha)
print("Best l1_ratio:", best_l1_ratio)

y_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_pred)

print("Train R^2 (ElasticNet):", r2_train)

y_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_pred)

print("Test R^2 (ElasticNet):", r2_test)

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": model.named_steps["enet"].coef_
})

threshold = 1e-20
nonzero_df = coef_df[coef_df["coef"].abs() > threshold]
print(nonzero_df.sort_values("coef", ascending=False))


# ------------ Ridge Regression ------------------

alpha_candidates =np.logspace(1,6,20)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(
        alphas=alpha_candidates, 
                      cv=tscv))
])


model.fit(X_train, y_train)

best_alpha = model.named_steps["ridge"].alpha_
print("Best alpha (Ridge):", best_alpha)

cv_r2_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha))
    ])

    model.fit(X_tr, y_tr)
    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    cv_r2_scores.append(r2)

    print(f"Fold {fold+1} R²: {r2:.6f}")

print("Mean CV R²:", np.mean(cv_r2_scores))

y_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_pred)

print("Train R^2 (Ridge):", r2_train)

y_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_pred)
print("Test R^2 (Ridge):", r2_test)

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": model.named_steps["ridge"].coef_
})

print(coef_df.sort_values("coef", ascending=False))


# ------------ Lasso Regression ------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(
        alphas=np.logspace(-5,-1,50),
        cv=tscv, random_state=0,max_iter=100000))
])
model.fit(X_train, y_train)

lasso = model.named_steps["lasso"]

# Retrieve the MSE path from LassoCV 
mse_path = lasso.mse_path_   # shape = (n_alpha, n_cv)

# Convert MSE to R²
y_var = np.var(y_train, ddof=1)
cv_r2 = 1 - mse_path / y_var   # shape (n_alpha, n_cv)

# Get the index of the best alpha
alpha_idx = list(lasso.alphas_).index(lasso.alpha_)

# Display the R² values for all CV splits at the best alpha
print("R² for each CV split at the best alpha:")
for i, r2 in enumerate(cv_r2[alpha_idx]):
    print(f"  CV{i+1}: {r2:.6f}")

best_alpha = model.named_steps["lasso"].alpha_
print("Best alpha (LassoCV):", best_alpha)

y_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_pred)

print("Train R^2 (LassoCV):", r2_train)

y_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_pred)

print("Test R^2(LassoCV):", r2_test)
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": model.named_steps["lasso"].coef_
})

threshold = 1e-20
nonzero_df = coef_df[coef_df["coef"].abs() > threshold]
zero_df = coef_df[coef_df["coef"].abs() <= threshold]
print(nonzero_df.sort_values("coef", ascending=False))
print(zero_df.sort_values("coef", ascending=False))