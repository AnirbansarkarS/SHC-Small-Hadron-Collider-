"""Particle model for basic 2D motion."""

import numpy as np


class Particle:
    def __init__(self, charge, mass, position, velocity):
        self.q = charge
        self.m = mass
        self.r = np.array(position, dtype=float)
        self.v = np.array(velocity, dtype=float)

    def kinetic_energy(self):
        return 0.5 * self.m * np.dot(self.v, self.v)
