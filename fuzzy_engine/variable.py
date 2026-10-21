"""
Linguistic Variable definition for Fuzzy Logic Systems.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from .membership import MembershipFunction


class LinguisticVariable:
    """
    Represents a fuzzy linguistic variable (e.g., 'Temperature', 'Fan Speed').
    
    Contains a universe of discourse [min_val, max_val] and a dictionary of
    linguistic terms, each mapped to a MembershipFunction.
    """

    def __init__(
        self,
        name: str,
        universe: Tuple[float, float],
        unit: str = "",
        resolution: int = 500,
        description: str = "",
    ):
        if universe[0] >= universe[1]:
            raise ValueError(f"Universe bounds must have min < max, got {universe}")
        self.name = name
        self.universe_min = float(universe[0])
        self.universe_max = float(universe[1])
        self.unit = unit
        self.resolution = resolution
        self.description = description
        self.terms: Dict[str, MembershipFunction] = {}

    def add_term(self, term_name: str, mf: MembershipFunction) -> "LinguisticVariable":
        """Attach a linguistic term and its membership function."""
        self.terms[term_name] = mf
        return self

    def fuzzify(self, crisp_value: float) -> Dict[str, float]:
        """
        Compute membership degrees for all terms at a given crisp value.
        Clamps value within universe boundaries.
        """
        clamped = float(np.clip(crisp_value, self.universe_min, self.universe_max))
        return {
            term_name: float(mf.evaluate(clamped))
            for term_name, mf in self.terms.items()
        }

    def get_universe_array(self) -> np.ndarray:
        """Return the discrete 1D grid of points covering the universe of discourse."""
        return np.linspace(self.universe_min, self.universe_max, self.resolution)

    def get_term_array(self, term_name: str) -> np.ndarray:
        """Return the membership function values evaluated across the entire universe grid."""
        if term_name not in self.terms:
            raise KeyError(f"Term '{term_name}' not defined for variable '{self.name}'")
        x = self.get_universe_array()
        return np.asarray(self.terms[term_name].evaluate(x), dtype=float)

    def __repr__(self) -> str:
        terms_str = ", ".join(self.terms.keys())
        return f"LinguisticVariable('{self.name}', universe=[{self.universe_min}, {self.universe_max}] {self.unit}, terms=[{terms_str}])"
