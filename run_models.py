"""
Task 3 — Forecasting Models
Models: SARIMA (statistical), LSTM (neural), TCN (neural)
Prediction target: Dec 16-22, 2013  (test week)
Training data: Nov 1 — Dec 15, 2013
"""
import warnings, gc, json, time, sys
warnings.filterwarnings("ignore")
from pathlib import Path

def flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

import numpy as np
import pandas as pd
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

BASE      = Path(".")
PROC      = BASE / "processed"
FIGS      = BASE / "figures"
RESULTS   = BASE / "results"
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ── Load target areas ──────────────────────────────────────────────────────────
meta = json.loads((PROC / "target_areas.json").read_text())
TARGET_AREAS = meta["areas"]
AREA_LABELS  = {meta["top_square"]: f"Area {meta['top_square']} (highest traffic)",
                4159: "Area 4159", 4556: "Area 4556"}

TRAIN_END = pd.Timestamp("2013-12-16")
TEST_START = pd.Timestamp("2013-12-16")
TEST_END   = pd.Timestamp("2013-12-23")

SEQ_LEN    = 144       # 1 day of history (144 × 10-min = 24 hours)
PERIOD     = 144       # one day

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flush(f"Device: {DEVICE}")

# ── Helper: build series ───────────────────────────────────────────────────────
def load_series(square_id):
    df = pd.read_parquet(PROC / "milan_internet_traffic.parquet",
                         engine="pyarrow",
                         filters=[("square_id", "=", square_id)])
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    s = df.set_index("datetime")["internet"].sort_index()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    return s.reindex(full_idx, fill_value=0.0)

# ── Metrics ────────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp   = mape(y_true, y_pred)
    return {"MAE": mae, "MAPE": mp, "RMSE": rmse}

# ── Sequence builder for neural models ────────────────────────────────────────
def make_sequences(values, seq_len):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: SARIMA
# ══════════════════════════════════════════════════════════════════════════════
def run_sarima(series, area_id):
    flush(f"\n[SARIMA] Area {area_id}")
    train = series[series.index < TRAIN_END]
    test  = series[(series.index >= TEST_START) & (series.index < TEST_END)]

    # Aggregate to hourly — period s=24 keeps SARIMA tractable (fast fit < 2 min)
    # Hourly resolution retains the dominant daily seasonality identified in Task 2.
    train_h = train.resample("1h").sum()
    test_h  = test.resample("1h").sum()

    flush(f"  Training samples (hourly): {len(train_h)}")
    t0 = time.time()
    # p=2,d=0,q=1 from PACF/ACF; P=1,D=1,Q=1,s=24 for daily seasonality
    model = SARIMAX(train_h, order=(2, 0, 1),
                    seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=60)
    train_time = time.time() - t0
    flush(f"  Fit complete in {train_time:.1f}s  AIC={fit.aic:.1f}")

    t1 = time.time()
    fc = fit.forecast(steps=len(test_h))
    infer_time = time.time() - t1

    # Upsample hourly forecast back to 10-min via linear interpolation
    fc_series = pd.Series(fc.values, index=test_h.index)
    fc_10 = fc_series.resample("10min").interpolate("linear")
    fc_10 = fc_10.reindex(test.index, method="nearest", fill_value=0.0)
    fc_10 = fc_10.clip(lower=0)

    metrics = eval_metrics(test.values, fc_10.values)
    metrics.update({"train_time": train_time, "infer_time": infer_time})
    flush(f"  MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE']:.2f}%  RMSE={metrics['RMSE']:.4f}")
    flush(f"  Train time: {train_time:.1f}s  Inference: {infer_time:.2f}s")
    return test, fc_10, metrics

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: LSTM
# ══════════════════════════════════════════════════════════════════════════════
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head  = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                   nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def run_lstm(series, area_id):
    flush(f"\n[LSTM] Area {area_id}")
    train_s = series[series.index < TRAIN_END].values
    test    = series[(series.index >= TEST_START) & (series.index < TEST_END)]

    scaler = MinMaxScaler()
    train_sc = scaler.fit_transform(train_s.reshape(-1,1)).flatten()

    X, y = make_sequences(train_sc, SEQ_LEN)
    # validation split: last 7 days of training
    val_cut = 7 * PERIOD
    X_tr, X_val = X[:-val_cut], X[-val_cut:]
    y_tr, y_val = y[:-val_cut], y[-val_cut:]

    ds_tr  = TensorDataset(torch.from_numpy(X_tr).unsqueeze(-1),
                           torch.from_numpy(y_tr))
    ds_val = TensorDataset(torch.from_numpy(X_val).unsqueeze(-1),
                           torch.from_numpy(y_val))
    dl_tr  = DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=128)

    model = LSTMForecaster(hidden=128, n_layers=2, dropout=0.2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.HuberLoss()

    best_val, patience_cnt, best_state = np.inf, 0, None
    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(1, 31):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(ds_tr)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl += loss_fn(model(xb), yb).item() * len(xb)
        vl /= len(ds_val)
        sched.step(vl)
        train_losses.append(ep_loss); val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= 5:
            flush(f"  Early stop at epoch {epoch}")
            break
        if epoch % 10 == 0:
            flush(f"  Epoch {epoch:3d} | train={ep_loss:.5f} val={vl:.5f}")
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()

    # Rolling one-step-ahead inference on test set
    seed  = train_sc[-SEQ_LEN:].copy()
    preds_sc = []
    t1 = time.time()
    with torch.no_grad():
        buf = seed.copy()
        for _ in range(len(test)):
            x = torch.tensor(buf[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            p = model(x).item()
            preds_sc.append(p)
            buf = np.append(buf, p)
    infer_time = time.time() - t1

    preds = scaler.inverse_transform(np.array(preds_sc).reshape(-1,1)).flatten()
    preds = np.clip(preds, 0, None)

    metrics = eval_metrics(test.values, preds)
    metrics.update({"train_time": train_time, "infer_time": infer_time})
    flush(f"  MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE']:.2f}%  RMSE={metrics['RMSE']:.4f}")
    flush(f"  Train time: {train_time:.1f}s  Inference: {infer_time:.2f}s")
    pred_series = pd.Series(preds, index=test.index)
    return test, pred_series, metrics, train_losses, val_losses

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: TCN (Temporal Convolutional Network)
# ══════════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        self._pad  = pad
    def forward(self, x):
        r = x
        out = self.relu(self.norm1(self.conv1(x)[:, :, :-self._pad] if self._pad else self.conv1(x)))
        out = self.drop(out)
        out = self.relu(self.norm2(self.conv2(out)[:, :, :-self._pad] if self._pad else self.conv2(out)))
        out = self.drop(out)
        return self.relu(out + r)

class TCN(nn.Module):
    def __init__(self, seq_len, channels=64, kernel=3, n_layers=5, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(1, channels, 1)
        dilations = [2**i for i in range(n_layers)]
        self.blocks = nn.Sequential(*[
            ResidualBlock(channels, kernel, d, dropout) for d in dilations
        ])
        self.head = nn.Linear(channels, 1)
    def forward(self, x):                 # x: (B, L, 1)
        x = x.permute(0, 2, 1)           # (B, 1, L)
        x = self.input_proj(x)            # (B, C, L)
        x = self.blocks(x)                # (B, C, L)
        x = x[:, :, -1]                   # (B, C) last step
        return self.head(x).squeeze(-1)

def run_tcn(series, area_id):
    flush(f"\n[TCN] Area {area_id}")
    train_s = series[series.index < TRAIN_END].values
    test    = series[(series.index >= TEST_START) & (series.index < TEST_END)]

    scaler  = MinMaxScaler()
    train_sc = scaler.fit_transform(train_s.reshape(-1,1)).flatten()

    X, y = make_sequences(train_sc, SEQ_LEN)
    val_cut = 7 * PERIOD
    X_tr, X_val = X[:-val_cut], X[-val_cut:]
    y_tr, y_val = y[:-val_cut], y[-val_cut:]

    ds_tr  = TensorDataset(torch.from_numpy(X_tr).unsqueeze(-1), torch.from_numpy(y_tr))
    ds_val = TensorDataset(torch.from_numpy(X_val).unsqueeze(-1), torch.from_numpy(y_val))
    dl_tr  = DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=128)

    model   = TCN(seq_len=SEQ_LEN, channels=64, kernel=3, n_layers=5, dropout=0.2).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.HuberLoss()

    best_val, patience_cnt, best_state = np.inf, 0, None
    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(1, 31):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(ds_tr)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                vl += loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        vl /= len(ds_val)
        sched.step(vl)
        train_losses.append(ep_loss); val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= 5:
            flush(f"  Early stop at epoch {epoch}")
            break
        if epoch % 10 == 0:
            flush(f"  Epoch {epoch:3d} | train={ep_loss:.5f} val={vl:.5f}")
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    seed = train_sc[-SEQ_LEN:].copy()
    preds_sc = []
    t1 = time.time()
    with torch.no_grad():
        buf = seed.copy()
        for _ in range(len(test)):
            x = torch.tensor(buf[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            p = model(x).item()
            preds_sc.append(p)
            buf = np.append(buf, p)
    infer_time = time.time() - t1

    preds = scaler.inverse_transform(np.array(preds_sc).reshape(-1,1)).flatten()
    preds = np.clip(preds, 0, None)
    metrics = eval_metrics(test.values, preds)
    metrics.update({"train_time": train_time, "infer_time": infer_time})
    flush(f"  MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE']:.2f}%  RMSE={metrics['RMSE']:.4f}")
    flush(f"  Train time: {train_time:.1f}s  Inference: {infer_time:.2f}s")
    pred_series = pd.Series(preds, index=test.index)
    return test, pred_series, metrics, train_losses, val_losses

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
all_results = {}
COLORS = {"SARIMA": "#e05c00", "LSTM": "#2a7fff", "TCN": "#22a86e"}

for area_id in TARGET_AREAS:
    flush(f"\n{'='*60}")
    flush(f"Processing Area {area_id}")
    flush('='*60)
    series = load_series(area_id)
    area_results = {}

    # SARIMA
    test_s, pred_sarima, met_sarima = run_sarima(series, area_id)
    area_results["SARIMA"] = {"pred": pred_sarima, "metrics": met_sarima}

    # LSTM
    test_l, pred_lstm, met_lstm, lstm_tr, lstm_vl = run_lstm(series, area_id)
    area_results["LSTM"] = {"pred": pred_lstm, "metrics": met_lstm,
                            "train_loss": lstm_tr, "val_loss": lstm_vl}

    # TCN
    test_t, pred_tcn, met_tcn, tcn_tr, tcn_vl = run_tcn(series, area_id)
    area_results["TCN"] = {"pred": pred_tcn, "metrics": met_tcn,
                           "train_loss": tcn_tr, "val_loss": tcn_vl}

    all_results[area_id] = {"area_results": area_results, "test": test_s}

    # ── Prediction plot (9 plots total, 3 per area) ─────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, (model_name, col) in zip(axes, COLORS.items()):
        pred = area_results[model_name]["pred"]
        ax.plot(test_s.index, test_s.values, color="black", lw=1.2,
                label="Actual", alpha=0.9)
        ax.plot(pred.index, pred.values, color=col, lw=1.2,
                ls="--", label=f"{model_name} forecast")
        m = area_results[model_name]["metrics"]
        ax.set_title(
            f"{model_name} — {AREA_LABELS.get(area_id, f'Area {area_id}')} | "
            f"MAE={m['MAE']:.3f}  MAPE={m['MAPE']:.1f}%  RMSE={m['RMSE']:.3f}",
            fontsize=10)
        ax.set_ylabel("Internet Activity")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[-1].set_xlabel("Date (Dec 2013)")
    fig.suptitle(f"Fig. — Prediction vs Actual | {AREA_LABELS.get(area_id, f'Area {area_id}')}", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS / f"predictions_area{area_id}.png", bbox_inches="tight")
    plt.close()
    flush(f"  Saved prediction plot for area {area_id}")

    # ── Metrics table ────────────────────────────────────────────────────────
    rows = []
    for mn in ["SARIMA", "LSTM", "TCN"]:
        m = area_results[mn]["metrics"]
        rows.append({"Model": mn, "MAE": round(m["MAE"],4),
                     "MAPE (%)": round(m["MAPE"],2), "RMSE": round(m["RMSE"],4)})
    tbl = pd.DataFrame(rows).set_index("Model")
    flush(f"\nMetrics table — Area {area_id}:")
    flush(tbl.to_string())
    tbl.to_csv(RESULTS / f"metrics_area{area_id}.csv")

    gc.collect()

# ── Timing summary ────────────────────────────────────────────────────────────
timing_rows = []
for mn in ["SARIMA", "LSTM", "TCN"]:
    tt_avg = np.mean([all_results[a]["area_results"][mn]["metrics"]["train_time"]
                      for a in TARGET_AREAS])
    it_avg = np.mean([all_results[a]["area_results"][mn]["metrics"]["infer_time"]
                      for a in TARGET_AREAS])
    timing_rows.append({"Model": mn,
                        "Avg Train Time (s)": round(tt_avg,1),
                        "Avg Inference Time (s)": round(it_avg,3)})

timing_df = pd.DataFrame(timing_rows).set_index("Model")
flush("\n=== Timing Summary ===")
flush(timing_df.to_string())
timing_df.to_csv(RESULTS / "timing_summary.csv")
flush("\nAll done.")
