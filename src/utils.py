import pandas as pd
import numpy as np
import yfinance as yf
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pandas_datareader import data as pdr
from tqdm import tqdm

def get_r(maturity = "DGS1", lookback_days = 14):
    end = dt.datetime.today()
    start = end - dt.timedelta(days = lookback_days)
    df = pdr.DataReader(maturity, "fred", start, end)
    r_annual = float(df.dropna().iloc[-1, 0]) / 100.0
    return (1 + r_annual) ** (1/252) - 1

def get_returns(tickers, start, end):
    if isinstance(tickers, str):
        tickers = [tickers]
    frames = []
    for t in tickers:
        df = yf.download(t, start = start, end = end, progress = False)
        if df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].squeeze().pct_change().dropna().rename(t)
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis = 1).dropna()

def parameters(df):
    mu = df.mean()
    sigma = df.cov()
    return mu, sigma

def optimize_weights(mu, Sigma, objective = "max_sharpe", r = 0.0, cutoff = 1e-10):
    n = len(mu)
    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    if objective == "min_var":
        def obj(w):
            return float(w @ Sigma.values @ w)
    elif objective == "max_sharpe":
        def obj(w):
            ret = float(w @ mu.values)
            vol = np.sqrt(float(w @ Sigma.values @ w))
            return - (ret - r) / (vol + 1e-12)
    else:
        raise ValueError("objective must be 'min_var' or 'max_sharpe'")
    res = minimize(obj, w0, method = "SLSQP", bounds = bounds, constraints = cons)
    w = res.x
    w[np.abs(w) < cutoff] = 0.0
    if w.sum() != 0:
        w = w / w.sum()
    return pd.Series(w, index = mu.index)

def evaluate(df, w, r = 0.0):
    periods_per_year = 252
    returns = (df * w).sum(axis = 1)
    mean = returns.mean() * periods_per_year
    vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
    r_ann = (1 + r) ** periods_per_year - 1
    sharpe = (mean - r_ann) / (vol + 1e-12)
    return {"mean_ann": float(mean), "vol_ann": float(vol), "sharpe_ann": float(sharpe)}

def monte_carlo(df, iterations = 500, objective = "max_sharpe", r = 0.0, seed = None):
    rng = np.random.default_rng(seed)
    n = len(df)
    tickers = df.columns
    W = np.empty((iterations, len(tickers)))
    for b in tqdm(range(iterations), desc = "Monte Carlo"):
        idx = rng.integers(0, n, size = n)
        sample = df.iloc[idx]
        mu = sample.mean()
        Sigma = sample.cov()
        w = optimize_weights(mu, Sigma, objective = objective, r = r)
        W[b, :] = w.values
    W = pd.DataFrame(W, columns = tickers)
    w = W.median(axis = 0)
    s = w.sum()
    if s != 0:
        w = w / s
    return W, w

def plot_weights(w_point, w_mc, title = "MVO vs Monte Carlo median"):
    tickers = w_point.index.tolist()
    x = np.arange(len(tickers))
    width = 0.38
    _, ax = plt.subplots(figsize = (10, 5))
    ax.bar(x - width/2, w_point.values, width, label = "MVO")
    ax.bar(x + width/2, w_mc.values, width, label = "MC median")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation = 45, ha = "right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()
    plt.show()

def plot_weight_bands(W, w_point = None, q_low = 5, q_high = 95, title = "Monte Carlo weight bands"):
    tickers = W.columns.tolist()
    p_low = np.percentile(W.values, q_low, axis = 0)
    p_high = np.percentile(W.values, q_high, axis = 0)
    p_med_raw = np.percentile(W.values, 50, axis = 0)
    s = p_med_raw.sum()
    p_med = p_med_raw / s if s != 0 else p_med_raw
    x = np.arange(len(tickers))
    _, ax = plt.subplots(figsize = (10, 5))
    ax.vlines(x, p_low, p_high, color = "tab:orange", linewidth = 6, alpha = 0.4, label = f"{q_low}–{q_high} % band")
    ax.scatter(x, p_med, color = "tab:orange", s = 50, label = "MC median", zorder = 3)
    if w_point is not None:
        ax.scatter(x, w_point.reindex(tickers).values, marker = "x", color = "tab:blue", s = 60, label = "MVO", zorder = 4)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation = 45, ha = "right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()
    plt.show()

def test(df_test, w_point, w_mc, r = 0.0):
    mvo = evaluate(df_test, w_point, r)
    mc  = evaluate(df_test, w_mc, r)
    return pd.DataFrame([mvo, mc], index = ["MVO", "MC"])
