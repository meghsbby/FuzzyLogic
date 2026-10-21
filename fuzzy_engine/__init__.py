"""
Fuzzy Logic Engine Package.
A pure, transparent Python/NumPy Mamdani fuzzy inference system.
"""

from .membership import (
    MembershipFunction,
    TriangularMF,
    TrapezoidalMF,
    GaussianMF,
    SigmoidMF,
)
from .variable import LinguisticVariable
from .rule import FuzzyRule, Rule, AntecedentClause, ConsequentClause
from .defuzzifier import defuzzify, DefuzzMethod
from .inference import MamdaniEngine, InferenceResult, RuleFiring

__all__ = [
    "MembershipFunction",
    "TriangularMF",
    "TrapezoidalMF",
    "GaussianMF",
    "SigmoidMF",
    "LinguisticVariable",
    "FuzzyRule",
    "Rule",
    "AntecedentClause",
    "ConsequentClause",
    "defuzzify",
    "DefuzzMethod",
    "MamdaniEngine",
    "InferenceResult",
    "RuleFiring",
]
