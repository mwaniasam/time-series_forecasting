"""
Task 3 — Forecasting Models: SARIMA, LSTM, TCN
Prediction target: Dec 16-22, 2013 (test week, one-step-ahead)
Training data:     Nov 1 — Dec 15, 2013
"""
import warnings, gc, json, time, sys
warnings.filterwarnings("ignore")
from pathlib import Path

def flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(42)
torch.manual_seed(42)

BASE    = Path(".")
PROC    = BASE / "processed"
FIGS    = BASE / "figures"
RESULTS = BASE / "results"
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

meta = json.loads((PROC / "target_areas.json").read_text())
TARGET_AREAS = meta["areas"]
AREA_LABELS = {
    meta["top_square"]: f"Area {meta['top_square']} (highest traffic)",
    4159: "Area 4159",
    4556: "Area 4556",
}

TRAIN_END  = pd.Timestamp("2013-12-16")
TEST_START = pd.Timestamp("2013-12-16")
TEST_END   = pd.Timestamp("2013-12-23")
SEQ_LEN    = 144   # 1 day of 10-min intervals
PERIOD     = 144   # one day
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flush(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_series(square_id):
    df = pd.read_parquet(
        PROC / "milan_internet_traffic.parquet",
        engine="pyarrow",
        filters=[("square_id", "=", square_id)],
    )
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    s = df.set_index("datetime")["internet"].sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    return s.reindex(idx, fill_value=0.0)


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def eval_metrics(y_true, y_pred):
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def make_sequences(values, seq_len):
    n = len(values) - seq_len
    X = np.lib.stride_tricks.as_strided(
        values,
        shape=(n, seq_len),
        strides=(values.strides[0], values.strides[0]),
    ).copy().astype(np.float32)
    y = values[seq_len:].astype(np.float32)
    return X, y


def fast_infer(model, seed, test_values, seq_len, device):
    """Fast one-step-ahead inference using actual test history."""
    combined = np.concatenate([seed[-seq_len:], test_values])
    X_test, _ = make_sequences(combined, seq_len)
    
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_test).unsqueeze(-1)), batch_size=256)
    preds = []
    with torch.no_grad():
        for xb in dl:
            preds.append(model(xb[0].to(device)).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# MODEL 1: SARIMA
# ---------------------------------------------------------------------------
def run_sarima(series, area_id):
    flush(f"\n[SARIMA] Area {area_id}")
    train = series[series.index < TRAIN_END]
    test  = series[(series.index >= TEST_START) & (series.index < TEST_END)]

    # Hourly mean aggregation: same magnitude as 10-min values, period s=24.
    train_h = train.resample("1h").mean()
    test_h  = test.resample("1h").mean()
    flush(f"  Training samples: {len(train_h)} hourly obs (s=24)")

    t0 = time.time()
    model = SARIMAX(
        train_h, order=(2, 0, 1), seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=60)
    train_time = time.time() - t0
    flush(f"  Fit done in {train_time:.1f}s  AIC={fit.aic:.1f}")

    t1 = time.time()
    fc = fit.forecast(steps=len(test_h))
    infer_time = time.time() - t1

    fc_series = pd.Series(fc.values, index=test_h.index)
    fc_10 = fc_series.resample("10min").interpolate("linear")
    fc_10 = fc_10.reindex(test.index, method="nearest", fill_value=0.0).clip(lower=0)

    metrics = eval_metrics(test.values, fc_10.values)
    metrics.update({"train_time": train_time, "infer_time": infer_time})
    flush(f"  MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE']:.2f}%  RMSE={metrics['RMSE']:.4f}")
    return test, fc_10, metrics


# ---------------------------------------------------------------------------
# MODEL 2: LSTM
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def run_neural(model_class, model_kwargs, model_name, series, area_id):
    flush(f"\n[{model_name}] Area {area_id}")
    train_s = series[series.index < TRAIN_END].values
    test    = series[(series.index >= TEST_START) & (series.index < TEST_END)]

    scaler   = MinMaxScaler()
    train_sc = scaler.fit_transform(train_s.reshape(-1, 1)).flatten().astype(np.float32)

    X, y     = make_sequences(train_sc, SEQ_LEN)
    val_cut  = 7 * PERIOD
    X_tr, X_val = X[:-val_cut], X[-val_cut:]
    y_tr, y_val = y[:-val_cut], y[-val_cut:]

    dl_tr  = DataLoader(TensorDataset(torch.from_numpy(X_tr).unsqueeze(-1),
                                       torch.from_numpy(y_tr)), batch_size=128, shuffle=True)
    dl_val = DataLoader(TensorDataset(torch.from_numpy(X_val).unsqueeze(-1),
                                       torch.from_numpy(y_val)), batch_size=256)

    model   = model_class(**model_kwargs).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.HuberLoss()

    best_val, pat, best_state = np.inf, 0, None
    t0 = time.time()

    for epoch in range(1, 31):
        model.train()
        ep_loss = sum(
            (lambda xb, yb: (opt.zero_grad(),
                              loss := loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)),
                              loss.backward(),
                              nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                              opt.step(),
                              loss.item() * len(xb))[-1])(xb, yb)
            for xb, yb in dl_tr
        ) / len(X_tr)

        model.eval()
        with torch.no_grad():
            vl = sum(
                loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
                for xb, yb in dl_val
            ) / len(X_val)
        sched.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1

        if pat >= 5:
            flush(f"  Early stop at epoch {epoch}  (best val={best_val:.5f})")
            break
        if epoch % 5 == 0:
            flush(f"  Epoch {epoch:3d}  train={ep_loss:.5f}  val={vl:.5f}")

    train_time = time.time() - t0
    flush(f"  Training done in {train_time:.1f}s")
    model.load_state_dict(best_state)

    t1 = time.time()
    test_sc = scaler.transform(test.values.reshape(-1, 1)).flatten().astype(np.float32)
    preds_sc = fast_infer(model, train_sc, test_sc, SEQ_LEN, DEVICE)
    infer_time = time.time() - t1

    preds = scaler.inverse_transform(preds_sc.reshape(-1, 1)).flatten().clip(0)
    metrics = eval_metrics(test.values, preds)
    metrics.update({"train_time": train_time, "infer_time": infer_time})
    flush(f"  MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE']:.2f}%  RMSE={metrics['RMSE']:.4f}")
    return test, pd.Series(preds, index=test.index), metrics


# ---------------------------------------------------------------------------
# MODEL 3: TCN
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
        )
        self._trim = pad * 2
        self.relu  = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        if self._trim:
            out = out[..., :-self._trim // 2]  # causal trim per conv
        # Ensure same length as residual
        out = out[..., :x.shape[-1]]
        return self.relu(out + x)


class TCN(nn.Module):
    def __init__(self, channels=64, kernel=3, n_layers=5, dropout=0.2):
        super().__init__()
        self.proj   = nn.Conv1d(1, channels, 1)
        self.blocks = nn.Sequential(*[
            _ResidualBlock(channels, kernel, 2 ** i, dropout)
            for i in range(n_layers)
        ])
        self.head = nn.Linear(channels, 1)

    def forward(self, x):            # x: (B, L, 1)
        x = x.permute(0, 2, 1)      # (B, 1, L)
        x = self.proj(x)             # (B, C, L)
        x = self.blocks(x)           # (B, C, L)
        return self.head(x[:, :, -1]).squeeze(-1)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
all_results = {}
COLORS = {"SARIMA": "#e05c00", "LSTM": "#2a7fff", "TCN": "#22a86e"}

for area_id in TARGET_AREAS:
    flush(f"\n{'='*60}")
    flush(f"Processing Area {area_id}")
    flush('=' * 60)
    series = load_series(area_id)
    area_res = {}

    test_s, pred_sarima, met_s = run_sarima(series, area_id)
    area_res["SARIMA"] = {"pred": pred_sarima, "metrics": met_s}

    test_l, pred_lstm, met_l = run_neural(
        LSTMForecaster, {"hidden": 128, "n_layers": 2, "dropout": 0.2},
        "LSTM", series, area_id,
    )
    area_res["LSTM"] = {"pred": pred_lstm, "metrics": met_l}

    test_t, pred_tcn, met_t = run_neural(
        TCN, {"channels": 64, "kernel": 3, "n_layers": 5, "dropout": 0.2},
        "TCN", series, area_id,
    )
    area_res["TCN"] = {"pred": pred_tcn, "metrics": met_t}

    all_results[area_id] = area_res

    # Prediction plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, (mn, col) in zip(axes, COLORS.items()):
        pred = area_res[mn]["pred"]
        ax.plot(test_s.index, test_s.values, color="black", lw=1.2, label="Actual")
        ax.plot(pred.index, pred.values, color=col, lw=1.2, ls="--", label=f"{mn} forecast")
        m = area_res[mn]["metrics"]
        ax.set_title(
            f"{mn} | {AREA_LABELS.get(area_id, f'Area {area_id}')} — "
            f"MAE={m['MAE']:.2f}  MAPE={m['MAPE']:.1f}%  RMSE={m['RMSE']:.2f}",
            fontsize=10,
        )
        ax.set_ylabel("Internet Activity")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Date (Dec 2013)")
    fig.suptitle(f"Predictions vs Actual — {AREA_LABELS.get(area_id, f'Area {area_id}')}", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS / f"predictions_area{area_id}.png", bbox_inches="tight", dpi=120)
    plt.close()
    flush(f"  Saved predictions_area{area_id}.png")

    # Per-area metrics CSV
    rows = [
        {"Model": mn, "MAE": round(m["MAE"], 4),
         "MAPE (%)": round(m["MAPE"], 2), "RMSE": round(m["RMSE"], 4)}
        for mn, m in [(k, area_res[k]["metrics"]) for k in ("SARIMA", "LSTM", "TCN")]
    ]
    tbl = pd.DataFrame(rows).set_index("Model")
    flush(f"\nMetrics — Area {area_id}:\n{tbl.to_string()}")
    tbl.to_csv(RESULTS / f"metrics_area{area_id}.csv")
    gc.collect()

# Timing summary
timing = pd.DataFrame([
    {
        "Model": mn,
        "Avg Train Time (s)": round(np.mean([all_results[a][mn]["metrics"]["train_time"] for a in TARGET_AREAS]), 1),
        "Avg Inference Time (s)": round(np.mean([all_results[a][mn]["metrics"]["infer_time"] for a in TARGET_AREAS]), 3),
    }
    for mn in ("SARIMA", "LSTM", "TCN")
]).set_index("Model")
flush(f"\n=== Timing Summary ===\n{timing.to_string()}")
timing.to_csv(RESULTS / "timing_summary.csv")
flush("\nAll done.")
