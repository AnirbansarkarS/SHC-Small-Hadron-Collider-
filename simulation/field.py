"""Magnetic field model for 2D motion."""

import numpy as np


class MagneticField:
    def __init__(self, B_z):
        self.B_z = B_z

    def lorentz_force(self, particle):
        vx, vy = particle.v
        Bz = self.B_z
        Fx = particle.q * vy * Bz
        Fy = -particle.q * vx * Bz
        return np.array([Fx, Fy])
