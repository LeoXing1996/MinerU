import numpy as np
from functools import partial
from scipy.integrate import quad


def integrand(t, a, b):
    return np.sqrt(a**2 * np.sin(t) ** 2 + b**2 * np.cos(t) ** 2)


class Line:
    def __init__(self, start, end, weight=1.0):
        self.start = start
        self.end = end
        self.weight = weight

    @property
    def length(self):
        return np.linalg.norm(self.end - self.start)


class Arc:
    def __init__(self, start, end, center, rx, ry, weight=1.0):
        self.start = start
        self.end = end
        self.center = center
        self.rx = rx
        self.ry = ry
        self.weight = weight

    @property
    def length(self):
        func = partial(integrand, a=self.rx, b=self.ry)
        return quad(func, self.start, self.end)[0]
