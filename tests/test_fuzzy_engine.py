"""
Unit tests for the custom Fuzzy Engine.
"""

import unittest
import numpy as np
from fuzzy_engine import (
    TriangularMF,
    TrapezoidalMF,
    GaussianMF,
    SigmoidMF,
    LinguisticVariable,
    Rule,
    AntecedentClause,
    ConsequentClause,
    MamdaniEngine,
    defuzzify,
)


class TestMembershipFunctions(unittest.TestCase):
    def test_triangular_mf(self):
        mf = TriangularMF(10, 20, 30)
        self.assertAlmostEqual(mf(10), 0.0)
        self.assertAlmostEqual(mf(15), 0.5)
        self.assertAlmostEqual(mf(20), 1.0)
        self.assertAlmostEqual(mf(25), 0.5)
        self.assertAlmostEqual(mf(30), 0.0)
        self.assertAlmostEqual(mf(5), 0.0)
        self.assertAlmostEqual(mf(35), 0.0)

    def test_trapezoidal_mf(self):
        mf = TrapezoidalMF(10, 20, 30, 40)
        self.assertAlmostEqual(mf(10), 0.0)
        self.assertAlmostEqual(mf(15), 0.5)
        self.assertAlmostEqual(mf(20), 1.0)
        self.assertAlmostEqual(mf(25), 1.0)
        self.assertAlmostEqual(mf(30), 1.0)
        self.assertAlmostEqual(mf(35), 0.5)
        self.assertAlmostEqual(mf(40), 0.0)

    def test_gaussian_mf(self):
        mf = GaussianMF(mean=20, sigma=5)
        self.assertAlmostEqual(mf(20), 1.0)
        self.assertAlmostEqual(mf(25), np.exp(-0.5))

    def test_sigmoid_mf(self):
        mf = SigmoidMF(c=20, a=1.0)
        self.assertAlmostEqual(mf(20), 0.5)
        self.assertGreater(mf(30), 0.9)
        self.assertLess(mf(10), 0.1)


class TestLinguisticVariable(unittest.TestCase):
    def test_fuzzify(self):
        temp = LinguisticVariable("Temperature", (0, 40), unit="°C")
        temp.add_term("Cold", TrapezoidalMF(0, 0, 10, 20))
        temp.add_term("Comfortable", TriangularMF(15, 22, 28))
        temp.add_term("Hot", TrapezoidalMF(25, 32, 40, 40))

        # At 22°C: Comfortable should be 1.0
        memberships = temp.fuzzify(22)
        self.assertAlmostEqual(memberships["Comfortable"], 1.0)
        self.assertAlmostEqual(memberships["Cold"], 0.0)
        self.assertAlmostEqual(memberships["Hot"], 0.0)

        # Clamping outside range
        memberships_high = temp.fuzzify(50)
        self.assertAlmostEqual(memberships_high["Hot"], 1.0)


class TestDefuzzifier(unittest.TestCase):
    def test_centroid_symmetric_triangle(self):
        x = np.linspace(0, 10, 1001)
        mf = TriangularMF(2, 5, 8).evaluate(x)
        cog = defuzzify(x, mf, method="centroid")
        self.assertAlmostEqual(cog, 5.0, places=2)

    def test_zero_activation_fallback(self):
        x = np.linspace(0, 100, 101)
        mf = np.zeros_like(x)
        cog = defuzzify(x, mf, method="centroid")
        self.assertAlmostEqual(cog, 50.0)


class TestMamdaniEngine(unittest.TestCase):
    def test_simple_inference(self):
        engine = MamdaniEngine(name="TestSystem")

        # Variable: Input
        temp = LinguisticVariable("Temp", (0, 40))
        temp.add_term("Cold", TrapezoidalMF(0, 0, 15, 22))
        temp.add_term("Hot", TrapezoidalMF(20, 28, 40, 40))
        engine.add_input(temp)

        # Variable: Output
        fan = LinguisticVariable("Fan", (0, 100))
        fan.add_term("Slow", TriangularMF(0, 25, 50))
        fan.add_term("Fast", TriangularMF(50, 75, 100))
        engine.add_output(fan)

        # Rules
        engine.add_rule(Rule(antecedents=[("Temp", "Cold")], consequents=[("Fan", "Slow")]))
        engine.add_rule(Rule(antecedents=[("Temp", "Hot")], consequents=[("Fan", "Fast")]))

        # Hot test
        res_hot = engine.compute({"Temp": 35})
        self.assertGreater(res_hot.crisp_outputs["Fan"], 60)
        self.assertEqual(len(res_hot.active_rules), 1)

        # Cold test
        res_cold = engine.compute({"Temp": 10})
        self.assertLess(res_cold.crisp_outputs["Fan"], 40)


if __name__ == "__main__":
    unittest.main()
