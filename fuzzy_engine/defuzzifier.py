"""
Defuzzification algorithms for converting aggregated fuzzy sets into crisp numbers.
"""

from typing import Literal
import numpy as np

DefuzzMethod = Literal["centroid", "bisector", "mom", "som", "lom"]


def defuzzify(
    x: np.ndarray,
    mf: np.ndarray,
    method: DefuzzMethod = "centroid",
) -> float:
    """
    Defuzzify an aggregated fuzzy membership curve.

    Parameters
    ----------
    x : np.ndarray
        1D universe of discourse points.
    mf : np.ndarray
        1D membership values corresponding to each point in x.
    method : DefuzzMethod
        'centroid': Center of Gravity (COG)
        'bisector': Divides the area under the curve into two equal halves
        'mom': Mean of Maximum
        'som': Smallest of Maximum
        'lom': Largest of Maximum

    Returns
    -------
    float: Crisp numerical value.
    """
    x = np.asarray(x, dtype=float)
    mf = np.asarray(mf, dtype=float)

    if x.shape != mf.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} and mf {mf.shape} must match")

    total_area = np.sum(mf)

    # Edge case: No rule fired or flat 0 everywhere -> return universe midpoint
    if total_area <= 1e-12:
        return float((x[0] + x[-1]) / 2.0)

    method = method.lower()  # type: ignore

    if method == "centroid":
        # Center of Gravity: sum(x * mf) / sum(mf)
        cog = np.sum(x * mf) / total_area
        return float(cog)

    elif method == "bisector":
        # Cumulative sum of area
        cumsum = np.cumsum(mf)
        half_area = total_area / 2.0
        idx = np.searchsorted(cumsum, half_area)
        if idx >= len(x):
            idx = len(x) - 1
        return float(x[idx])

    elif method in ("mom", "som", "lom"):
        max_val = np.max(mf)
        # Find all points close to max within numerical tolerance
        max_indices = np.where(np.isclose(mf, max_val, atol=1e-5))[0]

        if method == "mom":
            return float(np.mean(x[max_indices]))
        elif method == "som":
            return float(x[max_indices[0]])
        elif method == "lom":
            return float(x[max_indices[-1]])

    else:
        raise ValueError(f"Unknown defuzzification method '{method}'. Valid: 'centroid', 'bisector', 'mom', 'som', 'lom'")
