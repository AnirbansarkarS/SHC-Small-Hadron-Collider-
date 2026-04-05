"""Streamlit dashboard for the SHC simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model import predict_stability
from simulation.engine import run_simulation


@dataclass
class StabilityResult:
    label: str
    confidence: float


def _compute_drift(x: np.ndarray, y: np.ndarray) -> float:
    radius = np.sqrt(x**2 + y**2)
    mean_r = float(np.mean(radius))
    if mean_r == 0.0:
        return 1.0
    return float(np.max(np.abs(radius - mean_r)) / mean_r)


def _build_features(history: dict[str, list[float]], field_strength: float, initial_speed: float) -> pd.DataFrame:
    energy = np.asarray(history["energy"], dtype=float)
    drift = _compute_drift(
        np.asarray(history["x"], dtype=float),
        np.asarray(history["y"], dtype=float),
    )
    return pd.DataFrame(
        [
            {
                "field_strength": field_strength,
                "initial_speed": initial_speed,
                "drift": drift,
                "mean_energy": float(np.mean(energy)),
                "std_energy": float(np.std(energy)),
            }
        ]
    )

def _render_badge(result: StabilityResult) -> None:
    color = "#b91c1c" if result.label == "Unstable" else "#047857"
    st.markdown(
        (
            "<div style='display:inline-flex;align-items:center;gap:10px;"
            f"padding:8px 14px;border-radius:999px;background:{color};"
            "color:white;font-weight:600;'>"
            f"{result.label}"
            f"<span style='opacity:0.85;font-weight:500;'>"
            f"{result.confidence * 100:.1f}%"
            "</span></div>"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="SHC Live Dashboard", layout="wide")
    st.title("Small Hadron Collider: Live Simulation")
    st.caption("Adjust parameters to steer the particle and watch stability in real time.")

    with st.sidebar:
        st.header("Simulation controls")
        steps = st.slider("Steps", min_value=200, max_value=2500, value=1200, step=100)
        dt = st.number_input("Time step (s)", value=1e-9, format="%.1e")
        field_strength = st.slider("Magnetic field Bz (T)", 0.02, 0.3, 0.1, 0.005)
        initial_speed = st.slider("Initial speed (m/s)", 3e5, 3e6, 1e6, 1e5)
        charge = st.number_input("Charge (C)", value=1.6e-19, format="%.2e")
        mass = st.number_input("Mass (kg)", value=1.67e-27, format="%.2e")

        st.header("Stability model")
        st.info("Using pre-trained HIGGS model (Random Forest). Run ml/train.py to build it.")

    history = run_simulation(
        steps=steps,
        dt=dt,
        charge=charge,
        mass=mass,
        velocity=[initial_speed, 0.0],
        B_z=field_strength,
    )

    features = _build_features(history, field_strength, initial_speed)

    stability: StabilityResult | None = None
    with st.spinner("Running stability prediction..."):
        try:
            label, conf = predict_stability(history)
            stability = StabilityResult(label=label.capitalize(), confidence=conf / 100.0)
        except Exception as exc:  # pragma: no cover - guard for runtime issues
            st.error(f"Prediction error: {exc}")

    col1, col2 = st.columns([2, 1], gap="large")
    with col2:
        st.subheader("Stability prediction")
        if stability:
            _render_badge(stability)
        st.metric("Drift", f"{float(features['drift'].iloc[0]):.3f}")
        st.metric("Mean energy", f"{float(features['mean_energy'].iloc[0]):.3e} J")
        st.metric("Energy spread", f"{float(features['std_energy'].iloc[0]):.3e} J")

    with col1:
        st.subheader("Trajectory")
        traj_fig = _plot_trajectory(history)
        st.pyplot(traj_fig, clear_figure=True)

        st.subheader("Energy vs time")
        energy_fig = _plot_energy(history)
        st.pyplot(energy_fig, clear_figure=True)


def _plot_trajectory(history: dict[str, list[float]]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(history["x"], history["y"], lw=0.9, color="#1f77b4")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Particle trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _plot_energy(history: dict[str, list[float]]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(history["t"], history["energy"], lw=0.9, color="#ff7f0e")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Kinetic energy")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
