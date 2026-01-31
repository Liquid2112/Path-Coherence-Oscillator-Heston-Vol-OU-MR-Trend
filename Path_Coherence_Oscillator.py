"""
Path Coherence Oscillator (PCO) - Production Demo
=================================================

- Fetches daily data via yfinance
- Computes a rolling Path Coherence Oscillator
- Produces publication-quality Plotly charts:
  - Price + Oscillator panel
  - Scenario fan chart of future paths
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.stats import gaussian_kde
import warnings
from datetime import datetime, timedelta

import yfinance as yf  # pip install yfinance
import plotly.graph_objects as go  # pip install plotly
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# ------------------------------
# Core model: scenarios & PCO
# ------------------------------

@dataclass
class PathScenario:
    """
    Represents a single probabilistic future trajectory with metadata.
    """
    trajectory: np.ndarray          # Price path
    log_returns: np.ndarray
    volatility_path: np.ndarray     # For Heston-generated paths
    probability_weight: float       # Posterior weight from likelihood
    phase: float                    # Directional phase ([-pi/2, pi/2])
    features: dict                  # Structural features

    def calculate_geometry(self) -> dict:
        """Compute path geometry for alignment comparison."""
        # Curvature: L2 norm of second derivative
        curvature = np.gradient(np.gradient(self.trajectory))
        total_curvature = np.sqrt(np.sum(curvature**2))

        # Convexity: integrated second derivative sign
        convexity = np.sum(np.sign(np.gradient(np.gradient(self.trajectory))))

        # Trend consistency
        returns = np.diff(np.log(self.trajectory))
        trend_consistency = np.abs(np.mean(returns)) / (np.std(returns) + 1e-8)

        # Volatility trajectory shape
        vol_shape = np.std(np.diff(self.volatility_path))

        return {
            "curvature": total_curvature,
            "convexity": convexity,
            "trend_consistency": trend_consistency,
            "vol_shape": vol_shape,
            "terminal_return": self.log_returns[-1],
            "max_drawdown": np.min(
                self.trajectory / np.maximum.accumulate(self.trajectory) - 1.0
            ),
            "sharpe_path": np.mean(returns)
            / (np.std(returns) + 1e-8)
            * np.sqrt(252.0),
        }


class PathCoherenceOscillator:
    """
    Multi-Path Probability Oscillator using interference pattern analysis.

    Generates competing trajectory scenarios and measures their structural
    alignment through wave-like interference patterns.
    """

    def __init__(
        self,
        n_paths: int = 256,
        horizon: int = 20,  # Trading days ahead
        dt: float = 1.0 / 252.0,
        risk_free_rate: float = 0.04,
        # Heston parameters
        heston_kappa: float = 2.0,
        heston_theta: float = 0.04,
        heston_sigma: float = 0.3,
        heston_rho: float = -0.7,
        # OU parameters
        ou_theta: float = 0.2,
        ou_mu: float = 0.0,
        ou_sigma: float = 0.15,
        # Mixture weights
        trend_regime_prob: float = 0.4,
        meanrev_regime_prob: float = 0.4,
        random_walk_prob: float = 0.2,
    ):
        self.n_paths = n_paths
        self.horizon = horizon
        self.dt = dt
        self.r = risk_free_rate

        self.heston_params = {
            "kappa": heston_kappa,
            "theta": heston_theta,
            "sigma": heston_sigma,
            "rho": heston_rho,
        }

        self.ou_params = {
            "theta": ou_theta,
            "mu": ou_mu,
            "sigma": ou_sigma,
        }

        self.regime_weights = np.array(
            [trend_regime_prob, meanrev_regime_prob, random_walk_prob]
        )
        self.regime_weights /= self.regime_weights.sum()

        self.path_bundle: List[PathScenario] = []

    # ----- Path generators -----

    def _generate_heston_path(
        self,
        s0: float,
        v0: float,
        regime_strength: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Heston stochastic volatility model.

        dS = r S dt + sqrt(V) S dW1
        dV = kappa(theta - V) dt + sigma sqrt(V) dW2
        Corr(dW1, dW2) = rho
        """
        kappa = self.heston_params["kappa"] * regime_strength
        theta = self.heston_params["theta"]
        sigma_v = self.heston_params["sigma"]
        rho = self.heston_params["rho"]

        prices = np.zeros(self.horizon + 1)
        vols = np.zeros(self.horizon + 1)

        prices[0] = s0
        vols[0] = max(v0, 1e-6)

        dW1 = np.random.normal(0.0, np.sqrt(self.dt), self.horizon)
        dW2 = rho * dW1 + np.sqrt(1.0 - rho**2) * np.random.normal(
            0.0, np.sqrt(self.dt), self.horizon
        )

        for t in range(self.horizon):
            v_pos = max(vols[t], 0.0)

            vols[t + 1] = (
                vols[t]
                + kappa * (theta - v_pos) * self.dt
                + sigma_v * np.sqrt(v_pos) * dW2[t]
            )

            prices[t + 1] = prices[t] * np.exp(
                (self.r - 0.5 * v_pos) * self.dt + np.sqrt(v_pos) * dW1[t]
            )

        return prices, vols

    def _generate_ou_path(
        self,
        s0: float,
        target_price: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        OU mean-reverting path on log-prices.

        d log S = theta (mu - log S) dt + sigma dW
        """
        theta = self.ou_params["theta"]
        mu = self.ou_params["mu"]
        sigma = self.ou_params["sigma"]

        log_prices = np.zeros(self.horizon + 1)
        log_prices[0] = np.log(s0)

        if target_price is not None:
            mu = np.log(target_price)

        for t in range(self.horizon):
            dW = np.random.normal(0.0, np.sqrt(self.dt))
            log_prices[t + 1] = (
                log_prices[t] + theta * (mu - log_prices[t]) * self.dt + sigma * dW
            )

        prices = np.exp(log_prices)
        vols = np.ones_like(prices) * sigma
        return prices, vols

    def _generate_trend_path(
        self,
        s0: float,
        drift: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple trend-following GBM-style path."""
        rets = np.random.normal(drift * self.dt, 0.02 * np.sqrt(self.dt), self.horizon)
        log_prices = np.log(s0) + np.cumsum(np.insert(rets, 0, 0.0))
        prices = np.exp(log_prices)
        vols = np.ones_like(prices) * 0.15
        return prices, vols

    # ----- Bundle generation -----

    def generate_path_bundle(
        self,
        s0: float,
        v0: float,
        historical_vol: Optional[float] = None,
        recent_trend: Optional[float] = None,
    ) -> None:
        """Generate a mixture of regime paths from the current state."""
        self.path_bundle = []

        if recent_trend is not None and abs(recent_trend) > 0.5:
            adjusted_weights = np.array([0.6, 0.25, 0.15])
        else:
            adjusted_weights = self.regime_weights

        n_per_regime = (self.n_paths * adjusted_weights).astype(int)

        path_idx = 0

        # 1) Trend regime
        for _ in range(n_per_regime[0]):
            drift = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.15, 0.35) / 252.0
            if recent_trend is not None:
                drift = np.sign(recent_trend) * abs(drift)

            prices, vols = self._generate_trend_path(s0, drift)
            self._add_path(prices, vols, "trend", path_idx)
            path_idx += 1

        # 2) Mean-reversion (Heston + OU blend)
        for _ in range(n_per_regime[1]):
            prices_h, vols_h = self._generate_heston_path(s0, v0, regime_strength=1.5)
            target = s0 * (1.0 + np.random.normal(0.0, 0.05))
            prices_ou, vols_ou = self._generate_ou_path(s0, target)

            mix_w = np.random.beta(2.0, 2.0)
            prices = mix_w * prices_h + (1.0 - mix_w) * prices_ou
            vols = mix_w * vols_h + (1.0 - mix_w) * vols_ou

            self._add_path(prices, vols, "meanrev", path_idx)
            path_idx += 1

        # 3) Random / uncertain regime (mild Heston)
        for _ in range(n_per_regime[2]):
            prices, vols = self._generate_heston_path(s0, v0, regime_strength=0.5)
            self._add_path(prices, vols, "random", path_idx)
            path_idx += 1

    def _add_path(
        self,
        prices: np.ndarray,
        vols: np.ndarray,
        regime_type: str,
        idx: int,
    ) -> None:
        """Create PathScenario and add to bundle."""
        log_returns = np.diff(np.log(prices))

        path_vol = np.std(log_returns) * np.sqrt(252.0)
        if path_vol > 0.0:
            likelihood_weight = np.exp(-0.5 * (path_vol - 0.2) ** 2 / 0.1**2)
        else:
            likelihood_weight = 1.0

        terminal_return = np.sum(log_returns)
        vol_norm = np.std(log_returns) * np.sqrt(self.horizon) + 1e-8
        z = terminal_return / vol_norm
        phase = np.arctan(z)

        scenario = PathScenario(
            trajectory=prices,
            log_returns=log_returns,
            volatility_path=vols,
            probability_weight=likelihood_weight,
            phase=phase,
            features={"regime_type": regime_type, "path_id": idx},
        )
        self.path_bundle.append(scenario)

    # ----- Interference + oscillator -----

    def _compute_phase_concentration(
        self, phases: np.ndarray, weights: np.ndarray
    ) -> float:
        R = np.abs(np.sum(weights * np.exp(1j * phases))) / np.sum(weights)
        return float(R)

    def _find_peaks(self, density: np.ndarray, threshold: float = 0.3) -> List[int]:
        peaks = []
        for i in range(1, len(density) - 1):
            if (
                density[i] > density[i - 1]
                and density[i] > density[i + 1]
                and density[i] > threshold * density.max()
            ):
                peaks.append(i)
        return peaks

    def compute_interference_pattern(self) -> dict:
        """Compute interference/coherence metrics from the path ensemble."""
        if not self.path_bundle:
            raise ValueError("Path bundle is empty; call generate_path_bundle first.")

        amplitudes = np.array([p.probability_weight for p in self.path_bundle])
        phases = np.array([p.phase for p in self.path_bundle])

        complex_sum = np.sum(amplitudes * np.exp(1j * phases))
        total_weight = np.sum(amplitudes)
        coherence = np.abs(complex_sum) ** 2 / (total_weight**2 + 1e-10)

        geometries = [p.calculate_geometry() for p in self.path_bundle]
        feature_matrix = np.array(
            [
                [
                    g["curvature"],
                    g["trend_consistency"],
                    g["vol_shape"],
                    g["sharpe_path"],
                ]
                for g in geometries
            ]
        )

        norm_features = (feature_matrix - feature_matrix.mean(axis=0)) / (
            feature_matrix.std(axis=0) + 1e-8
        )
        feature_dispersion = np.trace(np.cov(norm_features.T))
        structural_alignment = 1.0 / (1.0 + feature_dispersion)

        terminal_returns = np.array([p.log_returns[-1] for p in self.path_bundle])
        kde = gaussian_kde(terminal_returns, weights=amplitudes)
        grid = np.linspace(terminal_returns.min(), terminal_returns.max(), 100)
        density = kde(grid)

        peaks = self._find_peaks(density)
        entropy = -np.sum(density * np.log(density + 1e-10)) / len(density)
        max_entropy = np.log(len(density))
        convergence_score = 1.0 - (entropy / max_entropy)

        return {
            "interference_coherence": float(coherence),
            "structural_alignment": float(structural_alignment),
            "terminal_convergence": float(convergence_score),
            "feature_dispersion": float(feature_dispersion),
            "n_modal_peaks": int(len(peaks)),
            "phase_concentration": self._compute_phase_concentration(
                phases, amplitudes
            ),
            "wave_amplitude": float(np.abs(complex_sum) / (total_weight + 1e-10)),
            "wave_phase": float(np.angle(complex_sum)),
        }

    def compute_oscillator(self) -> float:
        """
        Composite oscillator in [0, 100].

        80–100: high coherence (stable regime)
        50–80: mixed signals
        0–50: decoherence (regime transition zone)
        """
        pattern = self.compute_interference_pattern()
        osc = (
            0.35 * pattern["interference_coherence"]
            + 0.35 * pattern["structural_alignment"]
            + 0.20 * pattern["terminal_convergence"]
            + 0.10 * pattern["phase_concentration"]
        )
        return float(np.clip(osc, 0.0, 1.0) * 100.0)


# ------------------------------
# Data + rolling evaluation
# ------------------------------

def load_data_yfinance(symbol: str, years_back: int = 3) -> pd.DataFrame:
    """Download daily data for a symbol using yfinance.

    Ensures the DataFrame always has an `AdjClose` column for downstream
    processing (handles variants 'Adj Close' or falls back to 'Close').
    """
    end = datetime.today()
    start = end - timedelta(days=365 * years_back)
    df = yf.download(symbol, start=start, end=end, interval="1d")  # daily data
    df = df.dropna()

    # Robustly ensure an `AdjClose` column exists
    if "AdjClose" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "AdjClose"})
        elif "Close" in df.columns:
            df["AdjClose"] = df["Close"].copy()
        else:
            raise ValueError("Downloaded data missing 'Close' or 'Adj Close' columns")

    return df


def compute_realized_vol(log_returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized variance (per day)."""
    # Realized variance per day
    var = log_returns.rolling(window).var()
    return var


def compute_recent_trend(log_returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling trend score: cumulative return / realized vol."""
    cum_ret = log_returns.rolling(window).sum()
    vol = log_returns.rolling(window).std() * np.sqrt(window)
    trend_score = cum_ret / (vol + 1e-8)
    return trend_score


def rolling_pco(
    prices: pd.Series,
    realized_var: pd.Series,
    trend_score: pd.Series,
    horizon: int = 20,
    n_paths: int = 256,
    step: int = 5,  # evaluate every N days for speed
) -> pd.Series:
    """
    Compute PCO on a rolling basis.

    Returns a Series aligned to price dates (where PCO is evaluated).
    """
    pco = PathCoherenceOscillator(n_paths=n_paths, horizon=horizon)
    osc_values = {}
    idxs = range(horizon * 2, len(prices) - horizon, step)

    for i in idxs:
        date = prices.index[i]
        s0 = float(prices.iloc[i])
        v0 = float(realized_var.iloc[i])
        recent = float(trend_score.iloc[i])

        if np.isnan(v0) or v0 <= 0.0:
            continue

        pco.generate_path_bundle(s0=s0, v0=v0, recent_trend=recent)
        osc = pco.compute_oscillator()
        osc_values[date] = osc

    osc_series = pd.Series(osc_values).sort_index()
    return osc_series


# ------------------------------
# Visualization with Plotly
# ------------------------------

def plot_price_and_pco(df: pd.DataFrame, pco_series: pd.Series, symbol: str) -> go.Figure:
    """
    Create a 2-row subplot: price + PCO oscillator.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"{symbol} Price", "Path Coherence Oscillator"),
    )  # [web:53]

    # Top: price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["AdjClose"],
            name=f"{symbol} Adj Close",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1,
        col=1,
    )

    # Optional moving average for context
    ma = df["AdjClose"].rolling(50).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ma,
            name="50D MA",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Bottom: oscillator
    fig.add_trace(
        go.Scatter(
            x=pco_series.index,
            y=pco_series.values,
            name="PCO",
            line=dict(color="#2ca02c", width=2),
        ),
        row=2,
        col=1,
    )

    # Regime bands
    for y, name, color, dash in [
        (50, "Transition threshold", "#7f7f7f", "dot"),
        (80, "High-coherence threshold", "#d62728", "dot"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=pco_series.index,
                y=[y] * len(pco_series),
                name=name,
                line=dict(color=color, width=1, dash=dash),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="PCO (0–100)", row=2, col=1, range=[0, 100])

    fig.update_layout(
        title=f"{symbol}: Path Coherence Oscillator",
        xaxis2=dict(title="Date"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=800,
    )

    return fig


def plot_scenario_fan(
    df: pd.DataFrame,
    pco: PathCoherenceOscillator,
    s0: float,
    v0: float,
    recent_trend: float,
    symbol: str,
    lookback_days: int = 60,
) -> go.Figure:
    """
    Plot last N days of history + scenario fan of future paths from today.
    """
    pco.generate_path_bundle(s0=s0, v0=v0, recent_trend=recent_trend)

    recent_df = df.iloc[-lookback_days:]
    start_date = recent_df.index[-1]
    future_dates = pd.bdate_range(start_date, periods=pco.horizon + 1)[1:]

    fig = go.Figure()

    # History
    fig.add_trace(
        go.Scatter(
            x=recent_df.index,
            y=recent_df["AdjClose"],
            name=f"{symbol} History",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # Future scenarios (faint)
    for scenario in pco.path_bundle:
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=scenario.trajectory[1:],
                mode="lines",
                line=dict(color="rgba(31,119,180,0.10)", width=1),
                showlegend=False,
            )
        )

    # Highlight mean path
    mean_path = np.mean([p.trajectory for p in pco.path_bundle], axis=0)
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=mean_path[1:],
            name="Mean Scenario",
            line=dict(color="#d62728", width=2),
        )
    )

    fig.update_layout(
        title=f"{symbol}: Scenario Fan from Current State",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=600,
    )

    return fig


# ------------------------------
# Main demo
# ------------------------------

def main():
    symbol = "SPY"  # change to TSLA, AAPL, etc.
    print(f"Downloading data for {symbol}...")
    df = load_data_yfinance(symbol, years_back=3)

    df["LogRet"] = np.log(df["AdjClose"]).diff()
    df.dropna(inplace=True)

    realized_var = compute_realized_vol(df["LogRet"], window=20)
    trend_score = compute_recent_trend(df["LogRet"], window=20)

    print("Computing rolling PCO (this may take a bit)...")
    pco_series = rolling_pco(
        prices=df["AdjClose"],
        realized_var=realized_var,
        trend_score=trend_score,
        horizon=20,
        n_paths=256,
        step=5,
    )

    # Price + oscillator chart
    fig_panel = plot_price_and_pco(df, pco_series, symbol)
    fig_panel.show()

    # Scenario fan from the latest date
    last_idx = df.index[-1]
    s0 = float(df["AdjClose"].iloc[-1])
    v0 = float(realized_var.iloc[-1])
    recent = float(trend_score.iloc[-1])

    pco = PathCoherenceOscillator(n_paths=256, horizon=20)
    fig_fan = plot_scenario_fan(df, pco, s0, v0, recent, symbol)
    fig_fan.show()


if __name__ == "__main__":
    main()
