"""
aqi_utils.py – shared functions for every station notebook.
Same steps as the Delhi notebook, written once so every station is processed the same way.
"""
import os
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

# ------------------------------------------------------------------ constants
ALL_POLS = ["PM2.5", "PM10", "NO", "NO2", "NOX", "NH3", "SO2", "CO", "OZONE", "BENZENE", "TOLUENE"]
AQI_POLS_ALL = ["PM2.5", "PM10", "NO2", "SO2", "NH3", "CO", "OZONE"]
MAX8H = ["CO", "OZONE"]
WEATHER_MEAN = ["AT", "RH", "SR", "wind_speed_10m", "surface_pressure", "cloud_cover"]
WEATHER_COLS = WEATHER_MEAN + ["RF", "WD_sin", "WD_cos"]
CALENDAR_COLS = ["month", "weekday", "doy_sin", "doy_cos"]

BP = {"PM2.5": [0, 30, 60, 90, 120, 250, 380],
      "PM10":  [0, 50, 100, 250, 350, 430, 510],
      "NO2":   [0, 40, 80, 180, 280, 400, 520],
      "SO2":   [0, 40, 80, 380, 800, 1600, 2400],
      "NH3":   [0, 200, 400, 800, 1200, 1800, 2400],
      "CO":    [0, 1, 2, 10, 17, 34, 51],
      "OZONE": [0, 50, 100, 168, 208, 748, 1000]}
IDX  = [0, 50, 100, 200, 300, 400, 500]
CATS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

SPLITS = {"train": ("2023-01-01", "2024-09-30"),
          "val":   ("2024-10-01", "2024-12-31"),
          "test":  ("2025-01-01", "2025-12-31"),
          "test2": ("2026-01-01", "2026-09-30")}

# ------------------------------------------------------------------ 1. load + tidy (Steps 2–3)
def load_cpcb(data_dir, header_row=10):
    """Read all CPCB .xlsx files in a folder into one hourly table."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".xlsx"))
    raw = pd.concat([pd.read_excel(os.path.join(data_dir, f), header=header_row) for f in files],
                    ignore_index=True)
    raw["time"] = pd.to_datetime(raw["From Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    raw = raw.dropna(subset=["time"]).drop(columns=["From Date", "To Date"])
    raw = raw.set_index("time").sort_index().apply(pd.to_numeric, errors="coerce")
    raw.columns = [c.strip().upper() if c.strip().upper() in ALL_POLS else c.strip() for c in raw.columns]
    print("files:", files)
    return raw

def tidy(raw):
    """Drop empty columns, remove duplicate hours, make every hour exist."""
    raw = raw.drop(columns=raw.columns[raw.isna().all()])
    raw = raw[~raw.index.duplicated(keep="first")]
    full = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    raw = raw.reindex(full)
    raw.index.name = "time"
    return raw

# ------------------------------------------------------------------ 2. clean (Step 4)
def remove_flatlines(s, hours=6):
    group = (s != s.shift()).cumsum()
    return s.mask(s.groupby(group).transform("size") >= hours)

def clean_pollutants(data):
    """Negative values, 6-hour flat lines, NO2/NH3 stuck at <= 0.1. Returns (data, report)."""
    pols = [p for p in ALL_POLS if p in data.columns]
    d = data.copy()
    rep = pd.DataFrame(index=pols)
    before = d[pols].notna().sum()
    d[pols] = d[pols].where(d[pols] >= 0)
    rep["negative"] = before - d[pols].notna().sum()
    before = d[pols].notna().sum()
    for p in pols:
        d[p] = remove_flatlines(d[p])
    rep["flatline"] = before - d[pols].notna().sum()
    before = d[pols].notna().sum()
    for p in [c for c in ["NO2", "NH3"] if c in pols]:
        d[p] = d[p].mask(d[p] <= 0.1)
    rep["stuck_0.1"] = before - d[pols].notna().sum()
    rep["missing_%_now"] = (d[pols].isna().mean() * 100).round(1)
    return d, rep

# ------------------------------------------------------------------ 3. ERA5 weather (Steps 7–9)
ERA5_VARS = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation", "precipitation",
             "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover"]

def get_era5(lat, lon, start, end, cache_file):
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": ",".join(ERA5_VARS), "models": "era5",
              "timezone": "Asia/Kolkata", "wind_speed_unit": "ms"}
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=180)
    r.raise_for_status()
    met = pd.DataFrame(r.json()["hourly"])
    met["time"] = pd.to_datetime(met["time"])
    met = met.set_index("time")
    met.to_csv(cache_file)
    return met

def use_era5(data, met):
    """Drop station weather, add ERA5 with the same column names as Delhi."""
    station_weather = [c for c in ["AT", "RH", "SR", "BP", "WD", "WS", "RF", "TOT-RF", "VWS"] if c in data.columns]
    era5 = met.rename(columns={"temperature_2m": "AT", "relative_humidity_2m": "RH",
                               "shortwave_radiation": "SR", "precipitation": "RF",
                               "wind_direction_10m": "WD"})
    return data.drop(columns=station_weather).join(era5)

# ------------------------------------------------------------------ 4. daily values + AQI (Step 12)
def daily_value(s, use_max8h, min_hours=16):
    enough = s.resample("D").count() >= min_hours
    val = s.rolling(8, min_periods=6).mean().resample("D").max() if use_max8h else s.resample("D").mean()
    return val.where(enough)

def make_daily_conc(data, pols):
    return pd.DataFrame({p: daily_value(data[p], p in MAX8H) for p in pols})

def sub_index(conc, p):
    c = conc.clip(lower=0, upper=BP[p][-1])
    return pd.Series(np.interp(c, BP[p], IDX), index=conc.index).where(conc.notna())

def compute_aqi(conc, pols=None):
    """CPCB AQI from daily concentrations (uses whichever AQI pollutants are in `pols`)."""
    pols = pols or [p for p in AQI_POLS_ALL if p in conc.columns]
    sub = pd.DataFrame({p: sub_index(conc[p], p) for p in pols})
    pm = [p for p in ["PM2.5", "PM10"] if p in sub.columns]
    ok = (sub.notna().sum(axis=1) >= 3) & sub[pm].notna().any(axis=1)
    return sub.max(axis=1).where(ok)

def to_cat(a):
    return pd.cut(a, [-1, 50, 100, 200, 300, 400, 10000], labels=CATS).astype(str)

def macro_f1(y, p):
    labels = sorted(set(y) | set(p))
    return f1_score(y, p, average="macro", labels=labels, zero_division=0)

# ------------------------------------------------------------------ 5. features (Step 14)
def build_features(data, daily_conc, pols):
    """Weather of the forecast day + pollution 1–3 days before + 7-day mean + calendar."""
    wx = data[WEATHER_MEAN].resample("D").mean()
    wx["RF"] = data["RF"].resample("D").sum()
    rad = np.deg2rad(data["WD"])
    wx["WD_sin"] = np.sin(rad).resample("D").mean()
    wx["WD_cos"] = np.cos(rad).resample("D").mean()
    X = wx.copy()
    for p in pols:
        for k in [1, 2, 3]:
            X[f"{p}_lag{k}"] = daily_conc[p].shift(k)
        X[f"{p}_mean7"] = daily_conc[p].shift(1).rolling(7, min_periods=5).mean()
    X["month"] = X.index.month
    X["weekday"] = X.index.dayofweek
    doy = X.index.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return X

def in_period(index, name):
    start, end = SPLITS[name]
    return (index >= start) & (index <= end)

# ------------------------------------------------------------------ 6. models (Steps 17–21)
MODELS = {"Lasso":      Lasso(alpha=0.01),
          "ExtraTrees": ExtraTreesRegressor(n_jobs=-1),
          "LightGBM":   LGBMRegressor(verbose=-1),
          "HistGB":     HistGradientBoostingRegressor()}
NEEDS_SCALING = ["Lasso"]
HANDLES_NAN   = ["HistGB", "LightGBM"]

def build_pipeline(name, params=None):
    model = clone(MODELS[name])
    if params:
        model.set_params(**params)
    steps = []
    if name not in HANDLES_NAN:
        steps.append(("fill", SimpleImputer(strategy="median")))
    if name in NEEDS_SCALING:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log1p, inverse_func=np.expm1)

def forecast(X_train, Y_train, X_pred, pols, name, params=None):
    """One model per pollutant: train on (X_train, Y_train), predict X_pred."""
    pred = pd.DataFrame(index=X_pred.index, columns=pols, dtype=float)
    for p in pols:
        ok = Y_train[p].notna()
        pred[p] = build_pipeline(name, params).fit(X_train[ok], Y_train.loc[ok, p]).predict(X_pred)
    return pred

# ------------------------------------------------------------------ 7. evaluation (Steps 16, 19, 22)
def scores(conc, days, Y, pers_conc, pols):
    """AQI accuracy + macro-F1 (same days as persistence) and PM2.5/PM10 skill vs persistence."""
    a_true, a_pred, a_pers = compute_aqi(Y, pols), compute_aqi(conc, pols), compute_aqi(pers_conc, pols)
    ok = days & a_true.notna() & a_pred.notna() & a_pers.notna()
    yt, yp = to_cat(a_true[ok]), to_cat(a_pred[ok])
    out = {"days": int(ok.sum()),
           "accuracy": round(accuracy_score(yt, yp), 3),
           "macro_F1": round(macro_f1(yt, yp), 3)}
    for p in ["PM2.5", "PM10"]:
        if p in pols:
            m = days & Y[p].notna() & conc[p].notna() & pers_conc[p].notna()
            out[f"{p}_skill"] = round(1 - mean_squared_error(Y.loc[m, p], conc.loc[m, p])
                                      / mean_squared_error(Y.loc[m, p], pers_conc.loc[m, p]), 3)
    return out

def skill_per_pollutant(pred, days, Y, pers_conc, pols):
    out = {}
    for p in pols:
        ok = days & Y[p].notna() & pred[p].notna() & pers_conc[p].notna()
        out[p] = round(1 - mean_squared_error(Y.loc[ok, p], pred.loc[ok, p])
                       / mean_squared_error(Y.loc[ok, p], pers_conc.loc[ok, p]), 2)
    return out

def fast_f1(y, p):
    f1s = []
    for c in np.union1d(y, p):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f1s.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.mean(f1s)

def gain_ci(seed_preds, days, Y, pers_conc, pols, n_boot=2000, block=7, seed=0):
    """Block-bootstrap 95% CI of accuracy and macro-F1 gain over persistence, averaged over seeds."""
    a_true, a_pers = compute_aqi(Y, pols), compute_aqi(pers_conc, pols)
    ok = days & a_true.notna() & a_pers.notna()
    for pr in seed_preds:
        ok &= compute_aqi(pr, pols).notna()
    d = Y.index[ok]
    code = {c: i for i, c in enumerate(CATS)}
    yt_all = to_cat(a_true[d]).map(code).values
    yq_all = to_cat(a_pers[d]).map(code).values
    ym_all = [to_cat(compute_aqi(pr, pols)[d]).map(code).values for pr in seed_preds]
    rng, n = np.random.default_rng(seed), len(d)
    g_acc, g_f1 = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        ix = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        yt, yq = yt_all[ix], yq_all[ix]
        acc_p, f1_p = (yq == yt).mean(), fast_f1(yt, yq)
        g_acc.append(np.mean([(ym[ix] == yt).mean() - acc_p for ym in ym_all]))
        g_f1.append(np.mean([fast_f1(yt, ym[ix]) - f1_p for ym in ym_all]))
    ci = lambda v: f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
    return {"days": n,
            "acc_gain": round(float(np.mean([(ym == yt_all).mean() for ym in ym_all]) - (yq_all == yt_all).mean()), 3),
            "acc_gain_95CI": ci(g_acc),
            "F1_gain": round(float(np.mean([fast_f1(yt_all, ym) for ym in ym_all]) - fast_f1(yt_all, yq_all)), 3),
            "F1_gain_95CI": ci(g_f1),
            "P(F1_gain>0)": round(float(np.mean(np.array(g_f1) > 0)), 3)}
# ===== Leakage tests: boundary errors (Mann–Whitney) and leaky vs day-ahead (McNemar) =====
def leak_tests(obs_cat, leaky_cat, our_cat, obs_aqi, station, out_dir="."):
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.contingency_tables import mcnemar

    obs_cat, leaky_cat, our_cat = map(np.asarray, (obs_cat, leaky_cat, our_cat))
    breaks = np.array([50, 100, 200, 300, 400])
    dist = np.abs(np.asarray(obs_aqi, dtype=float)[:, None] - breaks).min(axis=1)
    wrong = leaky_cat != obs_cat

    p_mw = mannwhitneyu(dist[wrong], dist[~wrong], alternative="less").pvalue

    a, b = leaky_cat == obs_cat, our_cat == obs_cat
    table = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
    p_mc = mcnemar(table, exact=True).pvalue

    pd.DataFrame({"station": station, "dist": dist, "wrong": wrong}).to_csv(
        f"{out_dir}/boundary_{station}.csv", index=False)

    print(f"{station}: errors n={wrong.sum()} | boundary Mann–Whitney p = {p_mw:.4f} "
          f"| McNemar leaky vs day-ahead p = {p_mc:.2e}")
    return {"station": station, "n_errors": int(wrong.sum()), "p_boundary": p_mw, "p_mcnemar": p_mc}


def leak_tests_pooled(out_dir="."):
    import glob
    from scipy.stats import mannwhitneyu
    allb = pd.concat([pd.read_csv(f) for f in glob.glob(f"{out_dir}/boundary_*.csv")])
    p = mannwhitneyu(allb.dist[allb.wrong], allb.dist[~allb.wrong], alternative="less").pvalue
    print(f"Pooled ({allb.station.nunique()} stations): errors n={allb.wrong.sum()} | "
          f"boundary Mann–Whitney p = {p:.2e}")
    return p