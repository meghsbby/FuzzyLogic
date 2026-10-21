"""
Mamdani Fuzzy Inference System Engine.

Executes end-to-end inference:
1. Fuzzification of crisp inputs.
2. Rule strength evaluation.
3. Implication (Clipping via min or scaling via prod).
4. Aggregation (Union via max).
5. Defuzzification (Centroid, Bisector, MOM, etc.).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
import numpy as np

from .variable import LinguisticVariable
from .rule import FuzzyRule
from .defuzzifier import defuzzify, DefuzzMethod


@dataclass
class RuleFiring:
    """Record of an activated rule and its strength."""
    index: int
    rule: FuzzyRule
    description: str
    firing_strength: float
    consequents: List[str]


@dataclass
class InferenceResult:
    """Detailed output and mathematical trace from an inference cycle."""
    crisp_inputs: Dict[str, float]
    fuzzified_inputs: Dict[str, Dict[str, float]]
    rule_firings: List[RuleFiring]
    crisp_outputs: Dict[str, float]
    output_distributions: Dict[str, np.ndarray]
    output_universes: Dict[str, np.ndarray]
    defuzzification_method: str

    @property
    def active_rules(self) -> List[RuleFiring]:
        """Rules that fired with non-zero activation strength."""
        return [r for r in self.rule_firings if r.firing_strength > 1e-4]


class MamdaniEngine:
    """
    Modular Mamdani Fuzzy Inference Engine.
    """

    def __init__(
        self,
        name: str = "FuzzyInferenceSystem",
        implication: Literal["min", "prod"] = "min",
        aggregation: Literal["max", "sum"] = "max",
        default_defuzz: DefuzzMethod = "centroid",
    ):
        self.name = name
        self.implication = implication
        self.aggregation = aggregation
        self.default_defuzz: DefuzzMethod = default_defuzz
        self.inputs: Dict[str, LinguisticVariable] = {}
        self.outputs: Dict[str, LinguisticVariable] = {}
        self.rules: List[FuzzyRule] = []

    def add_input(self, variable: LinguisticVariable) -> "MamdaniEngine":
        self.inputs[variable.name] = variable
        return self

    def add_output(self, variable: LinguisticVariable) -> "MamdaniEngine":
        self.outputs[variable.name] = variable
        return self

    def add_rule(self, rule: FuzzyRule) -> "MamdaniEngine":
        self.rules.append(rule)
        return self

    def add_rules(self, rules: List[FuzzyRule]) -> "MamdaniEngine":
        self.rules.extend(rules)
        return self

    def compute(
        self,
        crisp_inputs: Dict[str, float],
        defuzz_method: Optional[DefuzzMethod] = None,
    ) -> InferenceResult:
        """
        Execute full inference pipeline and return an InferenceResult.
        """
        method = defuzz_method or self.default_defuzz

        # 1. Fuzzification
        fuzzified: Dict[str, Dict[str, float]] = {}
        for var_name, var in self.inputs.items():
            val = crisp_inputs.get(var_name, (var.universe_min + var.universe_max) / 2.0)
            fuzzified[var_name] = var.fuzzify(val)

        # 2. Rule evaluation
        rule_firings: List[RuleFiring] = []
        rule_activations: List[tuple[FuzzyRule, float]] = []

        for idx, rule in enumerate(self.rules):
            strength = rule.evaluate_firing_strength(fuzzified)
            consequent_desc = [str(c) for c in rule.consequents]
            firing = RuleFiring(
                index=idx + 1,
                rule=rule,
                description=rule.description,
                firing_strength=strength,
                consequents=consequent_desc,
            )
            rule_firings.append(firing)
            if strength > 0.0:
                rule_activations.append((rule, strength))

        # 3. Implication & Aggregation for each output variable
        output_distributions: Dict[str, np.ndarray] = {}
        output_universes: Dict[str, np.ndarray] = {}
        crisp_outputs: Dict[str, float] = {}

        for out_name, out_var in self.outputs.items():
            u_grid = out_var.get_universe_array()
            output_universes[out_name] = u_grid
            agg_curve = np.zeros_like(u_grid, dtype=float)

            for rule, strength in rule_activations:
                for cons in rule.consequents:
                    if cons.variable == out_name:
                        term_curve = out_var.get_term_array(cons.term)
                        if self.implication == "min":
                            clipped = np.minimum(term_curve, strength)
                        elif self.implication == "prod":
                            clipped = term_curve * strength
                        else:
                            clipped = np.minimum(term_curve, strength)

                        if self.aggregation == "max":
                            agg_curve = np.maximum(agg_curve, clipped)
                        elif self.aggregation == "sum":
                            agg_curve = np.clip(agg_curve + clipped, 0.0, 1.0)

            output_distributions[out_name] = agg_curve

            # 4. Defuzzification
            crisp_val = defuzzify(u_grid, agg_curve, method=method)
            crisp_outputs[out_name] = round(crisp_val, 2)

        return InferenceResult(
            crisp_inputs=crisp_inputs,
            fuzzified_inputs=fuzzified,
            rule_firings=rule_firings,
            crisp_outputs=crisp_outputs,
            output_distributions=output_distributions,
            output_universes=output_universes,
            defuzzification_method=method,
        )
