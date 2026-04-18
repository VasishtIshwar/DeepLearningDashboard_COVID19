import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

st.set_page_config(
    page_title="COVID-19 Forecast Dashboard",
    page_icon="🦠",
    layout="wide",
)

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

class SEIAHR_Tuned(nn.Module):
    """Physics-Informed Neural Network (PINN) — SEIAHR compartmental model."""
    def __init__(self, N_pop, t_max_days, scaler_y):
        super().__init__()
        self.N = float(N_pop)
        self.t_max = float(t_max_days)
        self.scaler_y = scaler_y

        self.net_state = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 6),
        )
        self.net_beta = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

        self._psi_I = nn.Parameter(torch.tensor(-2.0))
        self._r     = nn.Parameter(torch.tensor(0.0))
        self._eta_A = nn.Parameter(torch.tensor(-1.0))
        self._k_c   = nn.Parameter(torch.tensor(0.0))

        self.fixed_sigma      = 1.0 / 3.0
        self.fixed_gamma      = 1.0 / 5.0
        self.fixed_epsilon    = 1.0 / 180.0
        self.importation_rate = 10.0 / self.N

    @property
    def sigma(self):   return torch.tensor(self.fixed_sigma,   device=self._psi_I.device)
    @property
    def gamma_A(self): return torch.tensor(self.fixed_gamma,   device=self._psi_I.device)
    @property
    def gamma_I(self): return torch.tensor(self.fixed_gamma,   device=self._psi_I.device)
    @property
    def gamma_H(self): return torch.tensor(self.fixed_gamma,   device=self._psi_I.device)
    @property
    def epsilon(self): return torch.tensor(self.fixed_epsilon, device=self._psi_I.device)

    @property
    def psi_I(self): return torch.sigmoid(self._psi_I)
    @property
    def r(self):     return torch.sigmoid(self._r)
    @property
    def eta_A(self): return torch.sigmoid(self._eta_A)
    @property
    def k_c(self):   return torch.sigmoid(self._k_c)

    def get_beta(self, t):
        return F.softplus(self.net_beta(t))

    def forward(self, t):
        return F.softmax(self.net_state(t), dim=1)

    def get_predicted_cases_normalized(self, t):
        y_frac = self.forward(t)
        E_frac = y_frac[:, 1:2]
        new_cases_frac  = self.k_c * self.r * self.sigma * E_frac
        new_cases_count = new_cases_frac * self.N
        min_v   = torch.tensor(self.scaler_y.data_min_[0], device=t.device)
        scale_v = torch.tensor(self.scaler_y.scale_[0],    device=t.device)
        return (new_cases_count - min_v) * scale_v

    def compute_physics_loss(self, t):
        t  = t.clone().requires_grad_(True)
        y  = self.forward(t)
        s, e, i_c, a, h, r_pop = (y[:, k:k+1] for k in range(6))
        scale = 1.0 / self.t_max

        def get_dt(var):
            return torch.autograd.grad(var, t, torch.ones_like(var), create_graph=True)[0]

        dsdt = get_dt(s)     * scale
        dedt = get_dt(e)     * scale
        didt = get_dt(i_c)   * scale
        dadt = get_dt(a)     * scale
        dhdt = get_dt(h)     * scale
        drdt = get_dt(r_pop) * scale

        beta_t     = self.get_beta(t)
        infectious = i_c + (self.eta_A * a)
        lambda_val = (beta_t * s * infectious) + self.importation_rate

        f_s = (self.epsilon * r_pop) - lambda_val
        f_e = lambda_val - (self.sigma * e)
        f_a = ((1 - self.r) * self.sigma * e)  - (self.gamma_A * a)
        f_i = (self.r * self.sigma * e)         - ((self.gamma_I + self.psi_I) * i_c)
        f_h = (self.psi_I * i_c)                - (self.gamma_H * h)
        f_r = (self.gamma_A * a) + (self.gamma_I * i_c) + (self.gamma_H * h) - (self.epsilon * r_pop)

        loss_ode = ((dsdt - f_s)**2 + (dedt - f_e)**2 + (dadt - f_a)**2 +
                    (didt - f_i)**2 + (dhdt - f_h)**2 + (drdt - f_r)**2).mean()

        beta_diff        = beta_t[1:] - beta_t[:-1]
        loss_beta_smooth = (beta_diff ** 2).mean()

        return loss_ode, loss_beta_smooth


class ImprovedLSTM(nn.Module):
    """Bidirectional LSTM with multi-head attention and residual connections."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 forecast_steps=1, dropout=0.2, bidirectional=True):
        super().__init__()
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.forecast_steps = forecast_steps
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        embed_dim = hidden_size * self.num_directions
        num_heads = min(4, embed_dim // 16)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.residual_linear = nn.Linear(input_size, embed_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_steps),
        )
        self.dropout   = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out    = self.layer_norm(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last         = attn_out[:, -1, :]
        residual     = self.residual_linear(x[:, -1, :])
        combined     = self.dropout(last + residual)
        return self.output_projection(combined)   # (batch, forecast_steps)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
POPULATION_MAP = {
    "United States": 331_000_000,
    "United Kingdom":  67_000_000,
    "Germany":         83_000_000,
    "France":          67_000_000,
    "India":        1_380_000_000,
    "Brazil":         214_000_000,
    "Australia":       25_000_000,
    "Canada":          38_000_000,
}
SEQ_LENGTH = 7   # LSTM look-back window (days)


# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data"
    "/master/public/data/owid-covid-data.csv"
)

@st.cache_data(show_spinner="📡 Fetching latest OWID COVID data…", ttl=3600)
def load_owid_data():
    """Download the OWID CSV directly from GitHub and cache it for 1 hour."""
    return pd.read_csv(OWID_URL, parse_dates=["date"])


def prepare_tensors(df, country, start, end, test_days, device):
    """Prepare normalised tensors for the PINN model."""
    country_col = "location" if "location" in df.columns else "country"
    mask = (
        (df[country_col] == country) &
        (df["date"] >= pd.Timestamp(start)) &
        (df["date"] <= pd.Timestamp(end))
    )
    sub = df[mask].sort_values("date").reset_index(drop=True)
    if len(sub) < test_days + 10:
        return None

    sub["new_cases_smoothed"] = (
        pd.to_numeric(sub["new_cases_smoothed"], errors="coerce").fillna(0)
    )
    raw_cases = sub["new_cases_smoothed"].values.reshape(-1, 1).astype(float)
    raw_days  = (sub["date"] - sub["date"].min()).dt.days.values.reshape(-1, 1).astype(float)

    scaler_t = MinMaxScaler()
    scaler_y = MinMaxScaler()
    t_norm   = scaler_t.fit_transform(raw_days)
    y_norm   = scaler_y.fit_transform(raw_cases)

    t_tensor = torch.tensor(t_norm, dtype=torch.float32, requires_grad=True).to(device)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32).to(device)

    n       = len(t_tensor)
    n_train = n - test_days
    return {
        "t": t_tensor, "y": y_tensor,
        "t_train": t_tensor[:n_train], "y_train": y_tensor[:n_train],
        "t_test":  t_tensor[n_train:], "y_test":  y_tensor[n_train:],
        "n_train": n_train, "dates": sub["date"],
        "scaler_t": scaler_t, "scaler_y": scaler_y,
        "t_max": raw_days.max(),
    }


def prepare_lstm_sequences(data, seq_length=SEQ_LENGTH):
    """
    Build sliding-window sequences from normalised case values.
    Returns train/test X/y tensors plus metadata.
    """
    y_np = data["y"].detach().cpu().numpy()   # (N, 1)
    n    = len(y_np)

    xs, ys = [], []
    for i in range(n - seq_length):
        xs.append(y_np[i : i + seq_length])      # (seq, 1)
        ys.append(y_np[i + seq_length, 0])        # scalar

    X = torch.tensor(np.array(xs), dtype=torch.float32)   # (N-seq, seq, 1)
    y = torch.tensor(np.array(ys), dtype=torch.float32)   # (N-seq,)

    # Align with PINN's n_train — sequences start seq_length days later
    n_seq_train = max(0, data["n_train"] - seq_length)

    return {
        "X_train": X[:n_seq_train],
        "y_train": y[:n_seq_train],
        "X_test":  X[n_seq_train:],
        "y_test":  y[n_seq_train:],
        "X_all":   X,
        "y_all":   y,
        "n_seq_train": n_seq_train,
        "seq_length":  seq_length,
    }


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_pinn(data, epochs, n_pop, device, progress_bar, status_text):
    model = SEIAHR_Tuned(n_pop, data["t_max"], data["scaler_y"]).to(device)

    params_beta = list(model.net_beta.parameters())
    params_rest = list(model.net_state.parameters()) + [
        model._psi_I, model._r, model._eta_A, model._k_c
    ]
    optimizer = torch.optim.Adam([
        {"params": params_rest, "lr": 1e-3},
        {"params": params_beta, "lr": 1e-4},
    ])

    t_phys = torch.linspace(0, 1, 1000).view(-1, 1).to(device).requires_grad_(True)
    ic_tgt = torch.tensor([[0.80, 0.0, 0.0, 0.0, 0.0, 0.20]], device=device)
    t0     = torch.zeros((1, 1), device=device)

    log = []
    for ep in range(epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_norm = model.get_predicted_cases_normalized(data["t_train"])
        loss_data = F.mse_loss(pred_norm, data["y_train"])
        loss_ode, loss_beta_smooth = model.compute_physics_loss(t_phys)
        loss_ic   = F.mse_loss(model(t0), ic_tgt)
        loss      = loss_data + 0.1 * loss_ode + 10.0 * loss_beta_smooth + 0.01 * loss_ic

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ep % max(1, epochs // 20) == 0:
            progress_bar.progress(ep / epochs)
            status_text.text(
                f"[PINN] Epoch {ep}/{epochs} | "
                f"Loss: {loss.item():.5f} | Data: {loss_data.item():.5f} | ODE: {loss_ode.item():.5f}"
            )
            log.append({"epoch": ep, "total": loss.item(),
                        "data": loss_data.item(), "ode": loss_ode.item()})

    progress_bar.progress(1.0)
    status_text.text("PINN training complete ✓")
    return model, pd.DataFrame(log)


def train_lstm(data, epochs, device, progress_bar, status_text):
    lstm_data = prepare_lstm_sequences(data)

    X_train = lstm_data["X_train"].to(device)
    y_train = lstm_data["y_train"].to(device)

    dataset = TensorDataset(X_train, y_train)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ImprovedLSTM(
        input_size=1, hidden_size=64, num_layers=2,
        forecast_steps=1, dropout=0.2, bidirectional=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    log = []
    for ep in range(epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            pred  = model(bx).squeeze(-1)
            loss  = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        scheduler.step(epoch_loss)

        if ep % max(1, epochs // 20) == 0:
            progress_bar.progress(ep / epochs)
            status_text.text(f"[LSTM] Epoch {ep}/{epochs} | Loss: {epoch_loss:.6f}")
            log.append({"epoch": ep, "total": epoch_loss,
                        "data": epoch_loss, "ode": 0.0})

    progress_bar.progress(1.0)
    status_text.text("LSTM training complete ✓")
    return model, pd.DataFrame(log), lstm_data


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate_pinn(model, data):
    model.eval()
    with torch.no_grad():
        pred_norm = model.get_predicted_cases_normalized(data["t"]).cpu().numpy()
        pred_all  = data["scaler_y"].inverse_transform(pred_norm).flatten()
        actual    = data["scaler_y"].inverse_transform(
            data["y"].detach().cpu().numpy()
        ).flatten()
    n    = data["n_train"]
    mae  = mean_absolute_error(actual[n:], pred_all[n:])
    mape = np.mean(np.abs((actual[n:] - pred_all[n:]) / (np.abs(actual[n:]) + 1))) * 100
    return pred_all, actual, mae, mape


def evaluate_lstm(lstm_model, lstm_data, data):
    """
    Run the LSTM over ALL sliding windows (train + test) using actual past values.
    Returns a pred array aligned with the original time axis (NaN for first seq_length days).
    """
    lstm_model.eval()
    seq_length = lstm_data["seq_length"]
    scaler_y   = data["scaler_y"]

    X_all = lstm_data["X_all"]  # (N-seq, seq, 1)
    with torch.no_grad():
        preds_norm = lstm_model(X_all).squeeze(-1).cpu().numpy()   # (N-seq,)

    # Inverse-transform each prediction
    preds_orig = scaler_y.inverse_transform(
        preds_norm.reshape(-1, 1)
    ).flatten()
    preds_orig = np.maximum(preds_orig, 0)

    # Pad the first seq_length days with NaN so the array is the same length as `actual`
    n_total  = len(data["y"])
    pred_all = np.full(n_total, np.nan)
    pred_all[seq_length:] = preds_orig

    actual = scaler_y.inverse_transform(
        data["y"].detach().cpu().numpy()
    ).flatten()

    n    = data["n_train"]
    # Only evaluate test segment (skip NaN-padded region)
    test_preds  = pred_all[n:]
    test_actual = actual[n:]
    valid_mask  = ~np.isnan(test_preds)
    mae  = mean_absolute_error(test_actual[valid_mask], test_preds[valid_mask])
    mape = np.mean(np.abs(
        (test_actual[valid_mask] - test_preds[valid_mask]) /
        (np.abs(test_actual[valid_mask]) + 1)
    )) * 100
    return pred_all, actual, mae, mape


def compute_hybrid(pinn_pred, lstm_pred):
    """
    Average PINN and LSTM predictions day-by-day.
    Where one is NaN (LSTM warm-up period), use the other.
    """
    hybrid = np.where(
        np.isnan(lstm_pred), pinn_pred,
        np.where(np.isnan(pinn_pred), lstm_pred,
                 (pinn_pred + lstm_pred) / 2.0)
    )
    return hybrid


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def forecast_chart(dates, actual, pinn_pred, lstm_pred, hybrid_pred, n_train, model_mode):
    fig = go.Figure()

    # ---- Training actual (always shown) ----
    fig.add_trace(go.Scatter(
        x=dates[:n_train], y=actual[:n_train],
        name="Training data", line=dict(color="#93c5fd", width=2), opacity=0.7,
    ))

    # ---- Test actual (always shown) ----
    fig.add_trace(go.Scatter(
        x=dates[n_train:], y=actual[n_train:],
        name="Actual (test)", line=dict(color="#2563eb", width=3),
    ))

    # ---- PINN traces ----
    if model_mode in ("PINN", "Hybrid"):
        fig.add_trace(go.Scatter(
            x=dates[:n_train], y=pinn_pred[:n_train],
            name="PINN fit", line=dict(color="#f97316", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=dates[n_train:], y=pinn_pred[n_train:],
            name="PINN forecast", line=dict(color="#dc2626", width=2),
        ))

    # ---- LSTM traces ----
    if model_mode in ("LSTM", "Hybrid"):
        # Drop the NaN warm-up at the front
        valid_idx = ~np.isnan(lstm_pred)
        lstm_dates = dates[valid_idx]
        lstm_vals  = lstm_pred[valid_idx]
        # Split into train / test portions
        train_mask_valid = np.array(range(len(dates)))[valid_idx] < n_train
        fig.add_trace(go.Scatter(
            x=lstm_dates[train_mask_valid],  y=lstm_vals[train_mask_valid],
            name="LSTM fit", line=dict(color="#a855f7", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=lstm_dates[~train_mask_valid], y=lstm_vals[~train_mask_valid],
            name="LSTM forecast", line=dict(color="#7c3aed", width=2),
        ))

    # ---- Hybrid trace ----
    if model_mode == "Hybrid":
        fig.add_trace(go.Scatter(
            x=dates[n_train:], y=hybrid_pred[n_train:],
            name="Hybrid forecast (avg)", line=dict(color="#16a34a", width=3),
        ))

    # Forecast-start line
    vline_x = str(pd.Timestamp(dates.iloc[n_train]).date())
    fig.add_shape(
        type="line", x0=vline_x, x1=vline_x, y0=0, y1=1, yref="paper",
        line=dict(dash="dot", color="gray", width=1),
    )
    fig.add_annotation(
        x=vline_x, y=1, yref="paper", text="Forecast start",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(color="gray", size=12),
    )

    fig.update_layout(
        title="COVID-19 Case Forecast",
        xaxis_title="Date", yaxis_title="New cases (7-day smoothed)",
        hovermode="x unified", height=460, margin=dict(t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def loss_chart(log_df, label=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_df["epoch"], y=log_df["total"],
        name="Total loss", line=dict(color="#7c3aed"),
    ))
    if log_df["data"].nunique() > 1:
        fig.add_trace(go.Scatter(
            x=log_df["epoch"], y=log_df["data"],
            name="Data loss", line=dict(color="#059669"),
        ))
    if log_df["ode"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=log_df["epoch"], y=log_df["ode"],
            name="ODE loss", line=dict(color="#d97706"),
        ))
    fig.update_layout(
        title=f"Training loss curves{' — ' + label if label else ''}",
        xaxis_title="Epoch", yaxis_title="Loss",
        yaxis_type="log", height=320, margin=dict(t=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def get_epi_params(model, device):
    model.eval()
    t_full = torch.linspace(0, 1, 1000).view(-1, 1).to(device)
    with torch.no_grad():
        beta_curve = model.get_beta(t_full).cpu().numpy().flatten()
    gamma_I = model.gamma_I.item()
    r0_curve = beta_curve / gamma_I
    return {
        "beta_curve": beta_curve, "r0_curve": r0_curve,
        "avg_beta": float(beta_curve.mean()), "min_beta": float(beta_curve.min()),
        "max_beta": float(beta_curve.max()),  "avg_r0":  float(r0_curve.mean()),
        "min_r0":  float(r0_curve.min()),     "max_r0":  float(r0_curve.max()),
        "sigma":   model.sigma.item(),         "gamma_I": gamma_I,
        "gamma_H": model.gamma_H.item(),       "r":       model.r.item(),
        "k_c":     model.k_c.item(),
    }


def compartment_chart(model, data, dates):
    model.eval()
    with torch.no_grad():
        y_frac = model(data["t"].detach()).cpu().numpy()
    labels = ["Susceptible", "Exposed", "Infectious (I)",
              "Asymptomatic (A)", "Hospitalised", "Recovered"]
    colors = ["#3b82f6", "#f59e0b", "#ef4444",
              "#f97316", "#8b5cf6", "#10b981"]
    fig = go.Figure()
    for k, (label, color) in enumerate(zip(labels, colors)):
        fig.add_trace(go.Scatter(x=dates, y=y_frac[:, k],
                                 name=label, line=dict(color=color, width=2)))
    fig.update_layout(
        title="SEIAHR compartments", xaxis_title="Date",
        yaxis_title="Fraction of population", hovermode="x unified",
        height=380, margin=dict(t=60),
    )
    return fig


def risk_recommendation(avg_r0):
    if avg_r0 >= 2.0:
        return ("🔴 High risk", "error",
                "R₀ > 2 — the virus is spreading rapidly. Strong precautions are advised: "
                "**wear a well-fitting mask in indoor spaces**, avoid crowded gatherings, "
                "and ensure good ventilation.")
    elif avg_r0 >= 1.0:
        return ("🟠 Moderate risk", "warning",
                "R₀ between 1 and 2 — the epidemic is still growing but at a slower pace. "
                "**Wearing a mask in busy indoor settings** and maintaining hand hygiene are "
                "sensible precautions.")
    elif avg_r0 >= 0.5:
        return ("🟡 Low risk", "warning",
                "R₀ below 1 — the epidemic is declining. Continue standard hygiene practices. "
                "Stay alert to any upward trend.")
    else:
        return ("🟢 Very low risk", "success",
                "R₀ is very low — the virus is barely spreading. Normal activity is generally "
                "safe. Keep following baseline public health guidance.")


# ─────────────────────────────────────────────
# WORLD HEATMAP
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_map_data(_df):
    """Aggregate OWID data to monthly per-country averages for the choropleth."""
    keep = ["date", "iso_code", "location", "new_cases_smoothed_per_million"]
    df_m = _df[[c for c in keep if c in _df.columns]].copy()
    df_m = df_m.dropna(subset=["iso_code"])
    # OWID uses "OWID_*" codes for regional aggregates — drop them
    df_m = df_m[~df_m["iso_code"].str.startswith("OWID_", na=False)]
    df_m["month"] = df_m["date"].dt.to_period("M").astype(str)
    agg = (
        df_m.groupby(["month", "iso_code", "location"], as_index=False)
        ["new_cases_smoothed_per_million"]
        .mean()
        .rename(columns={"new_cases_smoothed_per_million": "cases_per_million"})
    )
    agg["cases_per_million"] = agg["cases_per_million"].clip(lower=0).round(1)
    return agg.sort_values("month").reset_index(drop=True)


def world_heatmap_fig(df_map):
    # Cap colour scale at 95th percentile so a few outlier spikes don't wash out the map
    cap = float(df_map["cases_per_million"].quantile(0.95))
    cap = max(cap, 1.0)

    fig = px.choropleth(
        df_map,
        locations="iso_code",
        color="cases_per_million",
        hover_name="location",
        hover_data={"iso_code": False, "cases_per_million": ":.1f"},
        animation_frame="month",
        color_continuous_scale="Reds",
        range_color=[0, cap],
        labels={"cases_per_million": "Cases / million"},
        title="COVID-19 New Cases per Million — Monthly Average (7-day smoothed)",
    )
    fig.update_layout(
        height=620,
        margin=dict(t=70, l=0, r=0, b=0),
        coloraxis_colorbar=dict(
            title="Cases<br>per million",
            thickness=14,
            len=0.55,
            tickformat=",",
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#555",
            showland=True,
            landcolor="#1a1a2e",
            showocean=True,
            oceancolor="#0d1117",
            showlakes=False,
            projection_type="natural earth",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Tune animation speed: 450 ms per frame, 200 ms transition
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 450
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Model selection")
    model_mode = st.radio(
        "Select model",
        ["PINN", "LSTM", "Hybrid"],
        index=0,
        help=(
            "**PINN** — Physics-Informed Neural Network (SEIAHR).  \n"
            "**LSTM** — Bidirectional LSTM with attention.  \n"
            "**Hybrid** — Runs both models and averages their forecasts."
        ),
    )
    st.divider()

    st.subheader("Data selection")
    # Load data once here so the country dropdown is populated from the live feed
    _df_sidebar = load_owid_data()
    _country_col = "location" if "location" in _df_sidebar.columns else "country"
    _available   = sorted(_df_sidebar[_country_col].dropna().unique().tolist())
    _default_idx = _available.index("United States") if "United States" in _available else 0
    country  = st.selectbox("Country", _available, index=_default_idx)
    n_pop    = POPULATION_MAP.get(country, 10_000_000)
    start_date = st.date_input("Start date", value=pd.Timestamp("2023-01-01"))
    end_date   = st.date_input("End date",   value=pd.Timestamp("2023-06-01"))
    test_days  = st.slider("Forecast window (days)", 7, 60, 21)
    st.divider()

    st.subheader("Training")
    epochs = st.slider("Epochs", 500, 10_000, 3_000, step=500)
    if model_mode in ("LSTM", "Hybrid"):
        lstm_epochs = st.slider("LSTM epochs", 10, 200, 50, step=10)
    st.divider()

    st.subheader("Save / load model")
    save_path_pinn = st.text_input("PINN model file",  value="pinn_model.pt")
    save_path_lstm = st.text_input("LSTM model file",  value="lstm_model.pt")
    col1, col2 = st.columns(2)
    do_train   = col1.button("🚀 Train", use_container_width=True)
    do_load    = col2.button("📂 Load",  use_container_width=True)
    if do_load:
        st.session_state["load_requested"] = True


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.title("🦠 COVID-19 Forecasting Dashboard")
model_subtitle = {
    "PINN":   "Physics-Informed Neural Network (SEIAHR)",
    "LSTM":   "Bidirectional LSTM with Attention",
    "Hybrid": "Hybrid — PINN + LSTM (averaged forecast)",
}
st.caption(model_subtitle[model_mode])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.caption(f"Device: **{device}**")

df_full = load_owid_data()

tab_forecast, tab_map = st.tabs(["📈 Forecast", "🗺️ Map"])

# ══════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════
with tab_forecast:

    data = prepare_tensors(df_full, country, start_date, end_date, test_days, device)
    if data is None:
        st.error("Not enough data for this country / date range. Try a wider window.")
        st.stop()

    # ── Training ──────────────────────────────
    if do_train:
        pb   = st.progress(0.0)
        stat = st.empty()

        if model_mode in ("PINN", "Hybrid"):
            pinn_model, pinn_log = train_pinn(data, epochs, n_pop, device, pb, stat)
            st.session_state["pinn_model"]  = pinn_model
            st.session_state["pinn_log_df"] = pinn_log
            torch.save(pinn_model.state_dict(), save_path_pinn)

        if model_mode in ("LSTM", "Hybrid"):
            pb.progress(0.0)
            lstm_model, lstm_log, lstm_data_seqs = train_lstm(
                data, lstm_epochs, device, pb, stat
            )
            st.session_state["lstm_model"]     = lstm_model
            st.session_state["lstm_log_df"]    = lstm_log
            st.session_state["lstm_data_seqs"] = lstm_data_seqs
            torch.save(lstm_model.state_dict(), save_path_lstm)

        st.success("Training complete ✓  Models saved.")

    # ── Loading ───────────────────────────────
    if st.session_state.get("load_requested"):
        if model_mode in ("PINN", "Hybrid") and os.path.exists(save_path_pinn):
            m = SEIAHR_Tuned(n_pop, data["t_max"], data["scaler_y"]).to(device)
            m.load_state_dict(torch.load(save_path_pinn, map_location=device))
            st.session_state["pinn_model"]  = m
            st.session_state["pinn_log_df"] = None

        if model_mode in ("LSTM", "Hybrid") and os.path.exists(save_path_lstm):
            lstm_seqs = prepare_lstm_sequences(data)
            m_lstm = ImprovedLSTM(
                input_size=1, hidden_size=64, num_layers=2,
                forecast_steps=1, dropout=0.2, bidirectional=True,
            ).to(device)
            m_lstm.load_state_dict(torch.load(save_path_lstm, map_location=device))
            st.session_state["lstm_model"]     = m_lstm
            st.session_state["lstm_log_df"]    = None
            st.session_state["lstm_data_seqs"] = lstm_seqs

        st.session_state["load_requested"] = False
        st.success("Model(s) loaded ✓")

    # ── Guard: need at least one model ────────
    need_pinn = model_mode in ("PINN", "Hybrid")
    need_lstm = model_mode in ("LSTM", "Hybrid")

    if need_pinn and "pinn_model" not in st.session_state:
        st.warning("Train a PINN model or load one from the sidebar.")
        st.stop()
    if need_lstm and "lstm_model" not in st.session_state:
        st.warning("Train an LSTM model or load one from the sidebar.")
        st.stop()

    # ── Evaluation ────────────────────────────
    pinn_pred = lstm_pred = hybrid_pred = None
    actual     = None

    if need_pinn:
        pinn_model = st.session_state["pinn_model"]
        pinn_pred, actual, pinn_mae, pinn_mape = evaluate_pinn(pinn_model, data)

    if need_lstm:
        lstm_model     = st.session_state["lstm_model"]
        lstm_data_seqs = st.session_state["lstm_data_seqs"]
        lstm_pred, actual, lstm_mae, lstm_mape = evaluate_lstm(lstm_model, lstm_data_seqs, data)

    if model_mode == "Hybrid":
        hybrid_pred = compute_hybrid(pinn_pred, lstm_pred)
        n_tr = data["n_train"]
        hp_test = hybrid_pred[n_tr:]
        ac_test = actual[n_tr:]
        valid   = ~np.isnan(hp_test)
        hybrid_mae  = mean_absolute_error(ac_test[valid], hp_test[valid])
        hybrid_mape = np.mean(np.abs(
            (ac_test[valid] - hp_test[valid]) / (np.abs(ac_test[valid]) + 1)
        )) * 100

    n_train = data["n_train"]
    dates   = data["dates"]

    # ── KPI row ───────────────────────────────
    if model_mode == "PINN":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("PINN MAE",  f"{pinn_mae:,.0f} cases")
        k2.metric("PINN MAPE", f"{pinn_mape:.1f}%")
        k3.metric("Training days", str(n_train))
        k4.metric("Forecast window", f"{test_days} days")

    elif model_mode == "LSTM":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("LSTM MAE",  f"{lstm_mae:,.0f} cases")
        k2.metric("LSTM MAPE", f"{lstm_mape:.1f}%")
        k3.metric("Training days", str(n_train))
        k4.metric("Forecast window", f"{test_days} days")

    else:  # Hybrid
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("PINN MAE",   f"{pinn_mae:,.0f}")
        k2.metric("LSTM MAE",   f"{lstm_mae:,.0f}")
        k3.metric("Hybrid MAE", f"{hybrid_mae:,.0f}")
        k4.metric("PINN MAPE",  f"{pinn_mape:.1f}%")
        k5.metric("LSTM MAPE",  f"{lstm_mape:.1f}%")
        k6.metric("Hybrid MAPE",f"{hybrid_mape:.1f}%")

    # ── Main forecast chart ───────────────────
    st.divider()
    st.plotly_chart(
        forecast_chart(dates, actual, pinn_pred, lstm_pred, hybrid_pred, n_train, model_mode),
        use_container_width=True,
    )

    # ── Loss curves + compartments ────────────
    col_l, col_r = st.columns(2)

    with col_l:
        if model_mode in ("PINN", "Hybrid"):
            pinn_log = st.session_state.get("pinn_log_df")
            if pinn_log is not None:
                st.plotly_chart(loss_chart(pinn_log, "PINN"), use_container_width=True)
            else:
                st.info("PINN loss curves not available for loaded models.")
        if model_mode in ("LSTM", "Hybrid"):
            lstm_log = st.session_state.get("lstm_log_df")
            if lstm_log is not None:
                st.plotly_chart(loss_chart(lstm_log, "LSTM"), use_container_width=True)
            else:
                st.info("LSTM loss curves not available for loaded models.")

    with col_r:
        if need_pinn:
            st.plotly_chart(compartment_chart(pinn_model, data, dates), use_container_width=True)

    # ── PINN-specific: β / R₀ section ─────────
    if need_pinn:
        st.divider()
        st.subheader("📈 Transmission dynamics: β and R₀")

        epi = get_epi_params(pinn_model, device)
        risk_label, risk_type, risk_msg = risk_recommendation(epi["avg_r0"])
        st.markdown(f"### {risk_label}")
        if risk_type == "error":
            st.error(risk_msg)
        elif risk_type == "warning":
            st.warning(risk_msg)
        else:
            st.success(risk_msg)

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Avg β",   f"{epi['avg_beta']:.4f}")
        b2.metric("Min β",   f"{epi['min_beta']:.4f}")
        b3.metric("Max β",   f"{epi['max_beta']:.4f}")
        b4.metric("Avg R₀",  f"{epi['avg_r0']:.2f}")
        b5.metric("R₀ range", f"{epi['min_r0']:.2f} – {epi['max_r0']:.2f}")


        with st.expander("ℹ️ What do β and R₀ mean?"):
            st.markdown("""
**β (beta) — transmission rate**
The average contacts per day multiplied by transmission probability per contact.
A higher β means the virus spreads more easily. β varies over time as behaviour,
immunity, and variants change — that is why the model learns a *curve* rather than
a single number.

**R₀ — basic reproduction number**
The average number of people one infected person goes on to infect, calculated as **β ÷ γ**.
- **R₀ > 1** → epidemic is growing
- **R₀ = 1** → epidemic is stable
- **R₀ < 1** → epidemic is declining

The dashed red line marks the critical threshold of R₀ = 1.
""")

        with st.expander("🔬 Learned biological parameters"):
            params = {
                "Average β":                    epi["avg_beta"],
                "Min β":                        epi["min_beta"],
                "Max β":                        epi["max_beta"],
                "Average R₀":                   epi["avg_r0"],
                "Latent period (days)":         1 / epi["sigma"],
                "Infectious period (days)":     1 / epi["gamma_I"],
                "Hospital stay (days)":         1 / epi["gamma_H"],
                "Symptomatic rate (%)":         epi["r"] * 100,
                "Case detection rate (%)":      epi["k_c"] * 100,
                "Hospitalisation rate (ψ_I)":   pinn_model.psi_I.item(),
                "Asymptomatic infectivity (η_A)": pinn_model.eta_A.item(),
            }
            st.dataframe(
                pd.DataFrame.from_dict(params, orient="index", columns=["Value"])
                .style.format("{:.4f}")
            )

    # ── Download predictions ───────────────────
    st.divider()
    pred_df = pd.DataFrame({"date": dates.values, "actual_cases": actual})
    if pinn_pred is not None:
        pred_df["pinn_forecast"] = pinn_pred
    if lstm_pred is not None:
        pred_df["lstm_forecast"] = lstm_pred
    if hybrid_pred is not None:
        pred_df["hybrid_forecast"] = hybrid_pred
    pred_df["split"] = ["train"] * n_train + ["test"] * (len(dates) - n_train)

    st.download_button(
        "⬇️ Download predictions as CSV",
        data=pred_df.to_csv(index=False),
        file_name="covid_forecast.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════
# TAB 2 — WORLD HEATMAP
# ══════════════════════════════════════════════
with tab_map:
    st.subheader("🌍 Global COVID-19 Spread")
    st.caption(
        "Monthly average of 7-day smoothed new cases per million people. "
        "Use the ▶ play button or drag the slider to travel through time."
    )

    with st.spinner("Building world map…"):
        df_map = build_map_data(df_full)

    if df_map.empty:
        st.error("Could not build map data from the loaded dataset.")
    else:
        # ── Optional metric selector ─────────────────
        months_available = sorted(df_map["month"].unique())
        col_info, col_range = st.columns([2, 3])
        with col_info:
            st.metric("Countries / territories", df_map["iso_code"].nunique())
            st.metric("Months of data", len(months_available))
            st.metric(
                "Date range",
                f"{months_available[0]}  →  {months_available[-1]}",
            )
        with col_range:
            all_months = sorted(df_map["month"].unique())
            start_m, end_m = st.select_slider(
                "Restrict time range shown on map",
                options=all_months,
                value=(all_months[0], all_months[-1]),
            )

        df_filtered = df_map[
            (df_map["month"] >= start_m) & (df_map["month"] <= end_m)
        ]

        st.plotly_chart(
            world_heatmap_fig(df_filtered),
            use_container_width=True,
        )

        with st.expander("ℹ️ How to read this map"):
            st.markdown("""
- **Colour intensity** shows the average daily new cases per million people for that month,
  using the OWID 7-day smoothed series.
- **Colour scale** is capped at the 95th percentile so a handful of extreme outbreak peaks
  don't wash out the rest of the world.
- **Regional aggregates** (World, Europe, Asia, etc.) are excluded — only individual
  countries are shown.
- **White / grey countries** have no data reported for that month.
- Use the **▶ Play** button to animate, or drag the slider manually.
            """)
