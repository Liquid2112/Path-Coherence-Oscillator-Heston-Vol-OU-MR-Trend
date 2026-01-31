"""
Path Coherence Oscillator (PCO) - HTML EXPORT VERSION
=====================================================
Saves charts as HTML files instead of opening in browser
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.stats import gaussian_kde
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm

import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


@dataclass
class PathScenario:
    """Represents a single probabilistic future trajectory with metadata."""
    trajectory: np.ndarray
    log_returns: np.ndarray
    volatility_path: np.ndarray
    probability_weight: float
    phase: float
    features: dict

    def calculate_geometry(self) -> dict:
        """Compute path geometry for alignment comparison."""
        grad1 = np.gradient(self.trajectory)
        grad2 = np.gradient(grad1)

        total_curvature = np.sqrt(np.sum(grad2**2))
        convexity = np.sum(np.sign(grad2))

        returns = np.diff(np.log(self.trajectory + 1e-10))
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-8

        trend_consistency = np.abs(mean_ret) / std_ret
        vol_shape = np.std(np.diff(self.volatility_path))

        cummax = np.maximum.accumulate(self.trajectory)
        drawdown = self.trajectory / (cummax + 1e-10) - 1.0

        return {
            "curvature": total_curvature,
            "convexity": convexity,
            "trend_consistency": trend_consistency,
            "vol_shape": vol_shape,
            "terminal_return": self.log_returns[-1] if len(self.log_returns) > 0 else 0.0,
            "max_drawdown": np.min(drawdown),
            "sharpe_path": mean_ret / std_ret * np.sqrt(252.0),
        }


class PathCoherenceOscillator:
    """Multi-Path Probability Oscillator using interference pattern analysis."""

    def __init__(
        self,
        n_paths: int = 128,
        horizon: int = 20,
        dt: float = 1.0 / 252.0,
        risk_free_rate: float = 0.04,
        heston_kappa: float = 2.0,
        heston_theta: float = 0.04,
        heston_sigma: float = 0.3,
        heston_rho: float = -0.7,
        ou_theta: float = 0.2,
        ou_mu: float = 0.0,
        ou_sigma: float = 0.15,
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
        self._rng = np.random.default_rng()

    def _generate_heston_path(
        self,
        s0: float,
        v0: float,
        regime_strength: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Heston stochastic volatility model - VECTORIZED."""
        kappa = self.heston_params["kappa"] * regime_strength
        theta = self.heston_params["theta"]
        sigma_v = self.heston_params["sigma"]
        rho = self.heston_params["rho"]

        prices = np.empty(self.horizon + 1)
        vols = np.empty(self.horizon + 1)

        prices[0] = s0
        vols[0] = max(v0, 1e-6)

        sqrt_dt = np.sqrt(self.dt)
        dW1 = self._rng.normal(0.0, sqrt_dt, self.horizon)
        dW2 = rho * dW1 + np.sqrt(1.0 - rho**2) * self._rng.normal(0.0, sqrt_dt, self.horizon)

        for t in range(self.horizon):
            v_pos = max(vols[t], 0.0)

            vols[t + 1] = vols[t] + kappa * (theta - v_pos) * self.dt + sigma_v * np.sqrt(v_pos) * dW2[t]
            prices[t + 1] = prices[t] * np.exp((self.r - 0.5 * v_pos) * self.dt + np.sqrt(v_pos) * dW1[t])

        return prices, vols

    def _generate_ou_path(
        self,
        s0: float,
        target_price: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """OU mean-reverting path - VECTORIZED."""
        theta = self.ou_params["theta"]
        mu = self.ou_params["mu"]
        sigma = self.ou_params["sigma"]

        log_prices = np.empty(self.horizon + 1)
        log_prices[0] = np.log(s0)

        if target_price is not None:
            mu = np.log(target_price)

        sqrt_dt = np.sqrt(self.dt)
        for t in range(self.horizon):
            dW = self._rng.normal(0.0, sqrt_dt)
            log_prices[t + 1] = log_prices[t] + theta * (mu - log_prices[t]) * self.dt + sigma * dW

        prices = np.exp(log_prices)
        vols = np.full_like(prices, sigma)
        return prices, vols

    def _generate_trend_path(
        self,
        s0: float,
        drift: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple trend-following GBM-style path - VECTORIZED."""
        rets = self._rng.normal(drift * self.dt, 0.02 * np.sqrt(self.dt), self.horizon)
        log_prices = np.log(s0) + np.concatenate([[0.0], np.cumsum(rets)])
        prices = np.exp(log_prices)
        vols = np.full_like(prices, 0.15)
        return prices, vols

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
        n_per_regime[-1] = self.n_paths - n_per_regime[:-1].sum()

        path_idx = 0

        for _ in range(n_per_regime[0]):
            drift = self._rng.choice([-1.0, 1.0]) * self._rng.uniform(0.15, 0.35) / 252.0
            if recent_trend is not None:
                drift = np.sign(recent_trend) * abs(drift)

            prices, vols = self._generate_trend_path(s0, drift)
            self._add_path(prices, vols, "trend", path_idx)
            path_idx += 1

        for _ in range(n_per_regime[1]):
            prices_h, vols_h = self._generate_heston_path(s0, v0, regime_strength=1.5)
            target = s0 * (1.0 + self._rng.normal(0.0, 0.05))
            prices_ou, vols_ou = self._generate_ou_path(s0, target)

            mix_w = self._rng.beta(2.0, 2.0)
            prices = mix_w * prices_h + (1.0 - mix_w) * prices_ou
            vols = mix_w * vols_h + (1.0 - mix_w) * vols_ou

            self._add_path(prices, vols, "meanrev", path_idx)
            path_idx += 1

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
        log_returns = np.diff(np.log(prices + 1e-10))

        path_vol = np.std(log_returns) * np.sqrt(252.0)
        likelihood_weight = np.exp(-0.5 * (path_vol - 0.2) ** 2 / 0.1**2) if path_vol > 0.0 else 1.0

        terminal_return = np.sum(log_returns)
        vol_norm = np.std(log_returns) * np.sqrt(self.horizon) + 1e-8
        phase = np.arctan(terminal_return / vol_norm)

        scenario = PathScenario(
            trajectory=prices,
            log_returns=log_returns,
            volatility_path=vols,
            probability_weight=likelihood_weight,
            phase=phase,
            features={"regime_type": regime_type, "path_id": idx},
        )
        self.path_bundle.append(scenario)

    def _compute_phase_concentration(self, phases: np.ndarray, weights: np.ndarray) -> float:
        R = np.abs(np.sum(weights * np.exp(1j * phases))) / (np.sum(weights) + 1e-10)
        return float(R)

    def _find_peaks(self, density: np.ndarray, threshold: float = 0.3) -> List[int]:
        peaks = []
        if len(density) < 3:
            return peaks

        is_peak = (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
        is_significant = density[1:-1] > threshold * density.max()
        peak_indices = np.where(is_peak & is_significant)[0] + 1
        return peak_indices.tolist()

    def compute_interference_pattern(self) -> dict:
        """Compute interference/coherence metrics - OPTIMIZED."""
        if not self.path_bundle:
            raise ValueError("Path bundle is empty; call generate_path_bundle first.")

        amplitudes = np.array([p.probability_weight for p in self.path_bundle])
        phases = np.array([p.phase for p in self.path_bundle])

        complex_weights = amplitudes * np.exp(1j * phases)
        complex_sum = np.sum(complex_weights)
        total_weight = np.sum(amplitudes)
        coherence = np.abs(complex_sum) ** 2 / (total_weight**2 + 1e-10)

        geometries = [p.calculate_geometry() for p in self.path_bundle]
        feature_matrix = np.array([
            [g["curvature"], g["trend_consistency"], g["vol_shape"], g["sharpe_path"]]
            for g in geometries
        ])

        feature_mean = feature_matrix.mean(axis=0)
        feature_std = feature_matrix.std(axis=0) + 1e-8
        norm_features = (feature_matrix - feature_mean) / feature_std

        feature_dispersion = np.trace(np.cov(norm_features.T))
        structural_alignment = 1.0 / (1.0 + feature_dispersion)

        terminal_returns = np.array([p.log_returns[-1] if len(p.log_returns) > 0 else 0.0 
                                      for p in self.path_bundle])

        if len(np.unique(terminal_returns)) > 1:
            normalized_weights = amplitudes / (amplitudes.sum() + 1e-10)
            kde = gaussian_kde(terminal_returns, weights=normalized_weights)
            grid = np.linspace(terminal_returns.min(), terminal_returns.max(), 100)
            density = kde(grid)
            density = density / (density.sum() + 1e-10)
        else:
            density = np.ones(100) / 100.0

        peaks = self._find_peaks(density)
        entropy = -np.sum(density * np.log(density + 1e-10))
        max_entropy = np.log(len(density))
        convergence_score = 1.0 - (entropy / (max_entropy + 1e-10))

        return {
            "interference_coherence": float(coherence),
            "structural_alignment": float(structural_alignment),
            "terminal_convergence": float(convergence_score),
            "feature_dispersion": float(feature_dispersion),
            "n_modal_peaks": int(len(peaks)),
            "phase_concentration": self._compute_phase_concentration(phases, amplitudes),
            "wave_amplitude": float(np.abs(complex_sum) / (total_weight + 1e-10)),
            "wave_phase": float(np.angle(complex_sum)),
        }

    def compute_oscillator(self) -> float:
        """Composite oscillator in [0, 100]."""
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
    """Download daily data for a symbol using yfinance."""
    end = datetime.today()
    start = end - timedelta(days=365 * years_back)

    print(f"Fetching {symbol} data from {start.date()} to {end.date()}...")
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if "Adj Close" in df.columns:
        df["AdjClose"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["AdjClose"] = df["Close"]
    else:
        raise ValueError("Downloaded data missing Close columns")

    print(f"Loaded {len(df)} trading days")
    return df


def compute_realized_vol(log_returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized variance (per day)."""
    return log_returns.rolling(window).var()


def compute_recent_trend(log_returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling trend score: cumulative return / realized vol."""
    cum_ret = log_returns.rolling(window).sum()
    vol = log_returns.rolling(window).std() * np.sqrt(window)
    return cum_ret / (vol + 1e-8)


def rolling_pco(
    prices: pd.Series,
    realized_var: pd.Series,
    trend_score: pd.Series,
    horizon: int = 20,
    n_paths: int = 256,
    step: int = 3,
) -> pd.Series:
    """Compute PCO on a rolling basis with progress bar."""
    pco = PathCoherenceOscillator(n_paths=n_paths, horizon=horizon)
    osc_values = {}
    idxs = list(range(horizon * 2, len(prices) - horizon, step))

    print(f"\nComputing PCO for {len(idxs)} dates (n_paths={n_paths}, horizon={horizon})...")

    for i in tqdm(idxs, desc="PCO Calculation", unit="date"):
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
# Visualization - SAVE AS HTML
# ------------------------------

def plot_price_and_pco(df: pd.DataFrame, pco_series: pd.Series, symbol: str, 
                       save_path: str = "pco_panel.html") -> go.Figure:
    """Create a 2-row subplot: price + PCO oscillator and SAVE to HTML."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"{symbol} Price", "Path Coherence Oscillator"),
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["AdjClose"], name=f"{symbol} Adj Close",
        line=dict(color="#1f77b4", width=2)
    ), row=1, col=1)

    ma = df["AdjClose"].rolling(50).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=ma, name="50D MA",
        line=dict(color="#ff7f0e", width=1.5, dash="dash")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pco_series.index, y=pco_series.values, name="PCO",
        line=dict(color="#2ca02c", width=2)
    ), row=2, col=1)

    fig.add_hline(y=50, line_dash="dot", line_color="#7f7f7f", row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#d62728", row=2, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="PCO (0–100)", row=2, col=1, range=[0, 100])

    fig.update_layout(
        title=f"{symbol}: Path Coherence Oscillator",
        xaxis2=dict(title="Date"), template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=800,
    )

    # SAVE TO HTML
    fig.write_html(save_path)
    print(f"  ✓ Saved: {save_path}")

    return fig


def plot_scenario_fan(
    df: pd.DataFrame, pco: PathCoherenceOscillator,
    s0: float, v0: float, recent_trend: float,
    symbol: str, lookback_days: int = 60,
    save_path: str = "pco_scenario_fan.html"
) -> go.Figure:
    """Plot last N days of history + scenario fan and SAVE to HTML."""
    print("Generating scenario fan...")
    pco.generate_path_bundle(s0=s0, v0=v0, recent_trend=recent_trend)

    recent_df = df.iloc[-lookback_days:]
    start_date = recent_df.index[-1]
    future_dates = pd.bdate_range(start_date, periods=pco.horizon + 1)[1:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent_df.index, y=recent_df["AdjClose"],
        name=f"{symbol} History", line=dict(color="#1f77b4", width=2)
    ))

    # Sample paths for visualization
    sample_paths = pco.path_bundle[::max(1, len(pco.path_bundle)//50)]
    for scenario in sample_paths:
        fig.add_trace(go.Scatter(
            x=future_dates, y=scenario.trajectory[1:],
            mode="lines", line=dict(color="rgba(31,119,180,0.15)", width=1),
            showlegend=False
        ))

    mean_path = np.mean([p.trajectory for p in pco.path_bundle], axis=0)
    fig.add_trace(go.Scatter(
        x=future_dates, y=mean_path[1:],
        name="Mean Scenario", line=dict(color="#d62728", width=2)
    ))

    fig.update_layout(
        title=f"{symbol}: Scenario Fan from Current State",
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_white", height=600,
    )

    # SAVE TO HTML
    fig.write_html(save_path)
    print(f"  ✓ Saved: {save_path}")

    return fig


# ------------------------------
# Main demo
# ------------------------------

def main():
    symbol = "SPY"

    print("="*60)
    print("Path Coherence Oscillator - HTML Export Version")
    print("="*60)

    df = load_data_yfinance(symbol, years_back=3)

    print("\nCalculating features...")
    df["LogRet"] = np.log(df["AdjClose"]).diff()
    df.dropna(inplace=True)

    realized_var = compute_realized_vol(df["LogRet"], window=20)
    trend_score = compute_recent_trend(df["LogRet"], window=20)

    pco_series = rolling_pco(
        prices=df["AdjClose"],
        realized_var=realized_var,
        trend_score=trend_score,
        horizon=20,
        n_paths=256,
        step=3,
    )

    print(f"\n✓ PCO computed for {len(pco_series)} dates")
    print(f"  Latest reading: {pco_series.iloc[-1]:.1f}/100")

    # Generate and SAVE charts as HTML
    print("\nGenerating charts...")

    # Chart 1: Price + Oscillator Panel
    plot_price_and_pco(df, pco_series, symbol, save_path="pco_panel.html")

    # Chart 2: Scenario Fan
    s0 = float(df["AdjClose"].iloc[-1])
    v0 = float(realized_var.iloc[-1])
    recent = float(trend_score.iloc[-1])

    pco = PathCoherenceOscillator(n_paths=256, horizon=20)
    plot_scenario_fan(df, pco, s0, v0, recent, symbol, save_path="pco_scenario_fan.html")

    print("\n" + "="*60)
    print("✓ COMPLETE! Open these files in your browser:")
    print("  • pco_panel.html - Price & PCO Oscillator")
    print("  • pco_scenario_fan.html - Future Path Scenarios")
    print("="*60)


if __name__ == "__main__":
    main()