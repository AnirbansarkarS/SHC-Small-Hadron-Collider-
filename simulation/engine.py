"""Simple Lorentz-force simulation for a single particle."""

import matplotlib.pyplot as plt
import numpy as np

from simulation.field import MagneticField
from simulation.particle import Particle


def run_simulation(
    steps=1000,
    dt=1e-9,
    charge=1.6e-19,
    mass=1.67e-27,
    position=None,
    velocity=None,
    B_z=0.1,
):
    if position is None:
        position = [0.0, 0.0]
    if velocity is None:
        velocity = [1e6, 0.0]

    p = Particle(
        charge=charge,
        mass=mass,
        position=position,
        velocity=velocity,
    )
    field = MagneticField(B_z=B_z)

    history = {"t": [], "x": [], "y": [], "vx": [], "vy": [], "energy": []}

    a = field.lorentz_force(p) / p.m
    for step in range(steps):
        v_half = p.v + 0.5 * a * dt
        p.r += v_half * dt
        p.v = v_half
        a_new = field.lorentz_force(p) / p.m
        p.v = v_half + 0.5 * a_new * dt
        a = a_new

        history["t"].append((step + 1) * dt)
        history["x"].append(p.r[0])
        history["y"].append(p.r[1])
        history["vx"].append(p.v[0])
        history["vy"].append(p.v[1])
        history["energy"].append(p.kinetic_energy())

    return history


def plot_results(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["x"], history["y"], lw=0.8)
    ax1.set_title("Particle trajectory")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal")
    ax2.plot(history["energy"])
    ax2.set_title("Kinetic energy over time")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Energy (J)")
    plt.tight_layout()
    plt.savefig("analysis/trajectory.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    h = run_simulation()
    plot_results(h)
