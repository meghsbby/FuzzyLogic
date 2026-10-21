"""
Fuzzy Logic Membership Functions.

Provides standard continuous membership functions:
- Triangular (trimf)
- Trapezoidal (trapmf)
- Gaussian (gaussmf)
- Sigmoidal (sigmf)
- Generalized Bell (gbellmf)
"""

from abc import ABC, abstractmethod
import numpy as np


class MembershipFunction(ABC):
    """Abstract base class for all fuzzy membership functions."""

    @abstractmethod
    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the membership degree for a given input x."""
        pass

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.evaluate(x)


class TriangularMF(MembershipFunction):
    """
    Triangular Membership Function defined by 3 parameters: (a, b, c).
    
    a: left boundary (membership = 0)
    b: peak (membership = 1)
    c: right boundary (membership = 0)
    """

    def __init__(self, a: float, b: float, c: float, name: str = ""):
        if not (a <= b <= c):
            raise ValueError(f"Triangular parameters must satisfy a <= b <= c, got a={a}, b={b}, c={c}")
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.name = name

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y = np.zeros_like(x_arr)

        # Left slope: (x - a) / (b - a)
        if self.b > self.a:
            left_mask = (x_arr >= self.a) & (x_arr <= self.b)
            y[left_mask] = (x_arr[left_mask] - self.a) / (self.b - self.a)
        else:
            # Degenerate left shoulder (vertical wall at a == b)
            left_mask = x_arr <= self.b
            y[left_mask] = 1.0

        # Right slope: (c - x) / (c - b)
        if self.c > self.b:
            right_mask = (x_arr > self.b) & (x_arr <= self.c)
            y[right_mask] = (self.c - x_arr[right_mask]) / (self.c - self.b)
        else:
            # Degenerate right shoulder (vertical wall at b == c)
            right_mask = x_arr >= self.b
            y[right_mask] = 1.0

        # Peak exact match
        y[x_arr == self.b] = 1.0

        result = np.clip(y, 0.0, 1.0)
        return float(result) if np.isscalar(x) else result

    def __repr__(self) -> str:
        return f"TriangularMF(a={self.a}, b={self.b}, c={self.c})"


class TrapezoidalMF(MembershipFunction):
    """
    Trapezoidal Membership Function defined by 4 parameters: (a, b, c, d).
    
    a: left foot (membership = 0)
    b: left shoulder plateau start (membership = 1)
    c: right shoulder plateau end (membership = 1)
    d: right foot (membership = 0)
    """

    def __init__(self, a: float, b: float, c: float, d: float, name: str = ""):
        if not (a <= b <= c <= d):
            raise ValueError(f"Trapezoidal parameters must satisfy a <= b <= c <= d, got a={a}, b={b}, c={c}, d={d}")
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.name = name

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y = np.zeros_like(x_arr)

        # Left slope
        if self.b > self.a:
            left_mask = (x_arr >= self.a) & (x_arr < self.b)
            y[left_mask] = (x_arr[left_mask] - self.a) / (self.b - self.a)
        else:
            left_mask = x_arr <= self.b
            y[left_mask] = 1.0

        # Plateau
        plateau_mask = (x_arr >= self.b) & (x_arr <= self.c)
        y[plateau_mask] = 1.0

        # Right slope
        if self.d > self.c:
            right_mask = (x_arr > self.c) & (x_arr <= self.d)
            y[right_mask] = (self.d - x_arr[right_mask]) / (self.d - self.c)
        else:
            right_mask = x_arr >= self.c
            y[right_mask] = 1.0

        result = np.clip(y, 0.0, 1.0)
        return float(result) if np.isscalar(x) else result

    def __repr__(self) -> str:
        return f"TrapezoidalMF(a={self.a}, b={self.b}, c={self.c}, d={self.d})"


class GaussianMF(MembershipFunction):
    """
    Gaussian Membership Function defined by mean and standard deviation:
    
    mu(x) = exp( - (x - mean)^2 / (2 * sigma^2) )
    """

    def __init__(self, mean: float, sigma: float, name: str = ""):
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got sigma={sigma}")
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.name = name

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        val = np.exp(-0.5 * ((x_arr - self.mean) / self.sigma) ** 2)
        return float(val) if np.isscalar(x) else val

    def __repr__(self) -> str:
        return f"GaussianMF(mean={self.mean}, sigma={self.sigma})"


class SigmoidMF(MembershipFunction):
    """
    Sigmoidal Membership Function:
    
    mu(x) = 1 / (1 + exp(-a * (x - c)))
    
    c: center / inflection point
    a: slope parameter (positive for right-open, negative for left-open)
    """

    def __init__(self, c: float, a: float, name: str = ""):
        self.c = float(c)
        self.a = float(a)
        self.name = name

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        val = 1.0 / (1.0 + np.exp(-self.a * (x_arr - self.c)))
        return float(val) if np.isscalar(x) else val

    def __repr__(self) -> str:
        return f"SigmoidMF(c={self.c}, a={self.a})"
