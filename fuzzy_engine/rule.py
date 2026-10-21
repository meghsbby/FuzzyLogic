"""
Fuzzy Rule definitions and evaluation logic.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union


@dataclass
class AntecedentClause:
    """A single premise condition: e.g. 'Temperature IS Hot' or 'NOT Cold'."""
    variable: str
    term: str
    negated: bool = False

    def evaluate(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        var_memberships = fuzzified_inputs.get(self.variable, {})
        val = var_memberships.get(self.term, 0.0)
        return 1.0 - val if self.negated else val

    def __str__(self) -> str:
        prefix = "NOT " if self.negated else ""
        return f"{self.variable} IS {prefix}{self.term}"


@dataclass
class ConsequentClause:
    """A conclusion: e.g. 'FanSpeed IS Turbo'."""
    variable: str
    term: str

    def __str__(self) -> str:
        return f"{self.variable} IS {self.term}"


class FuzzyRule:
    """
    Mamdani Fuzzy Rule:
    IF (antecedents connected by AND/OR) THEN (consequents) [WITH weight]
    """

    def __init__(
        self,
        antecedents: List[AntecedentClause],
        consequents: List[ConsequentClause],
        operator: Literal["AND", "OR"] = "AND",
        t_norm: Literal["min", "prod"] = "min",
        s_norm: Literal["max", "prob_or"] = "max",
        weight: float = 1.0,
        description: Optional[str] = None,
    ):
        self.antecedents = antecedents
        self.consequents = consequents
        self.operator = operator.upper()
        self.t_norm = t_norm.lower()
        self.s_norm = s_norm.lower()
        self.weight = float(np.clip(weight, 0.0, 1.0)) if "np" in globals() else max(0.0, min(1.0, float(weight)))
        self.description = description or self._generate_description()

    def _generate_description(self) -> str:
        ant_str = f" {self.operator} ".join(str(a) for a in self.antecedents)
        con_str = " AND ".join(str(c) for c in self.consequents)
        w_str = f" (weight: {self.weight:.2f})" if self.weight < 1.0 else ""
        return f"IF {ant_str} THEN {con_str}{w_str}"

    def evaluate_firing_strength(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate firing strength (alpha) based on fuzzified input variables.
        """
        if not self.antecedents:
            return 0.0

        degrees = [ant.evaluate(fuzzified_inputs) for ant in self.antecedents]

        if self.operator == "AND":
            if self.t_norm == "min":
                raw_strength = min(degrees)
            elif self.t_norm == "prod":
                raw_strength = 1.0
                for d in degrees:
                    raw_strength *= d
            else:
                raw_strength = min(degrees)
        elif self.operator == "OR":
            if self.s_norm == "max":
                raw_strength = max(degrees)
            elif self.s_norm == "prob_or":
                # Probabilistic OR: a + b - a*b iteratively
                raw_strength = 0.0
                for d in degrees:
                    raw_strength = raw_strength + d - (raw_strength * d)
            else:
                raw_strength = max(degrees)
        else:
            raw_strength = min(degrees)

        # Apply rule weight
        firing_strength = raw_strength * self.weight
        return max(0.0, min(1.0, float(firing_strength)))

    def __repr__(self) -> str:
        return f"FuzzyRule({self.description})"


# Fluent builder helper for clean rule definitions
def Rule(
    antecedents: List[Union[Tuple[str, str], Tuple[str, str, bool], AntecedentClause]],
    consequents: List[Union[Tuple[str, str], ConsequentClause]],
    operator: Literal["AND", "OR"] = "AND",
    weight: float = 1.0,
    description: Optional[str] = None,
) -> FuzzyRule:
    """
    Convenience factory to quickly define rules with tuples:
    e.g. Rule(
        antecedents=[("Temp", "Hot"), ("Humidity", "Humid")],
        consequents=[("AC_Power", "High"), ("FanSpeed", "Turbo")]
    )
    """
    parsed_ants: List[AntecedentClause] = []
    for a in antecedents:
        if isinstance(a, AntecedentClause):
            parsed_ants.append(a)
        elif isinstance(a, tuple):
            if len(a) == 2:
                parsed_ants.append(AntecedentClause(variable=a[0], term=a[1], negated=False))
            elif len(a) == 3:
                parsed_ants.append(AntecedentClause(variable=a[0], term=a[1], negated=bool(a[2])))

    parsed_cons: List[ConsequentClause] = []
    for c in consequents:
        if isinstance(c, ConsequentClause):
            parsed_cons.append(c)
        elif isinstance(c, tuple):
            parsed_cons.append(ConsequentClause(variable=c[0], term=c[1]))

    return FuzzyRule(
        antecedents=parsed_ants,
        consequents=parsed_cons,
        operator=operator,
        weight=weight,
        description=description,
    )
