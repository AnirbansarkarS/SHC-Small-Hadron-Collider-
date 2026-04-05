"""Dataset generator for Phase 2."""

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.engine import run_simulation


def _add_noise(series: np.ndarray, std_frac: float, rng: np.random.Generator) -> np.ndarray:
    scale = np.std(series)
    if scale == 0.0:
        scale = 1.0
    return series + rng.normal(0.0, std_frac * scale, size=series.shape)


def _label_stability(x: np.ndarray, y: np.ndarray, drift_threshold: float):
    radius = np.sqrt(x**2 + y**2)
    mean_r = np.mean(radius)
    if mean_r == 0.0:
        return "unstable", 1.0
    drift = np.max(np.abs(radius - mean_r)) / mean_r
    label = "unstable" if drift > drift_threshold else "stable"
    return label, drift


def generate_dataset(
    output_path="data/datasets/phase2_runs.csv",
    runs=500,
    steps=1000,
    dt=1e-9,
    drift_threshold=0.05,
    noise_std=0.01,
    seed=42,
):
    rng = np.random.default_rng(seed)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_id in range(runs):
        B_z = rng.uniform(0.05, 0.2)
        speed = rng.uniform(5e5, 2e6)
        history = run_simulation(
            steps=steps,
            dt=dt,
            B_z=B_z,
            velocity=[speed, 0.0],
        )

        x = np.array(history["x"], dtype=float)
        y = np.array(history["y"], dtype=float)
        vx = np.array(history["vx"], dtype=float)
        vy = np.array(history["vy"], dtype=float)
        energy = np.array(history["energy"], dtype=float)
        time = np.array(history["t"], dtype=float)

        x = _add_noise(x, noise_std, rng)
        y = _add_noise(y, noise_std, rng)
        vx = _add_noise(vx, noise_std, rng)
        vy = _add_noise(vy, noise_std, rng)
        energy = _add_noise(energy, noise_std, rng)

        label, drift = _label_stability(x, y, drift_threshold)

        for idx in range(steps):
            rows.append(
                {
                    "run_id": run_id,
                    "time": time[idx],
                    "x": x[idx],
                    "y": y[idx],
                    "vx": vx[idx],
                    "vy": vy[idx],
                    "energy": energy[idx],
                    "field_strength": B_z,
                    "initial_speed": speed,
                    "drift": drift,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    generate_dataset()
