"""Regression checks for the exact-answer differentiation scorer."""
import unittest

from evaluate import grade, safe_expression


class GraderTests(unittest.TestCase):
    def test_equivalent_algebra(self):
        self.assertEqual(grade("Final answer: 2*(x + 1)", "2*x + 2"), (True, None))
        self.assertEqual(grade("Final answer: (2*x + 8 - 2*(2*x + 1))/(2*x + 8)**2",
                               "(6 - 2*x)/(2*x + 8)**2"), (True, None))

    def test_implicit_multiplication(self):
        self.assertEqual(grade("Final answer: 12x*cos(x**2)", "12*x*cos(x**2)"), (True, None))
        self.assertEqual(grade("Final answer: 3(x+1)(x-2)", "3*(x+1)*(x-2)"), (True, None))

    def test_missing_chain_rule_term(self):
        correct, _ = grade("Final answer: 12*cos(6*x**2 + 3*x + 7)",
                           "(12*x + 3)*cos(6*x**2 + 3*x + 7)")
        self.assertFalse(correct)

    def test_restricted_parser(self):
        for unsafe in ('__import__("os")', 'x.__class__', '100**100000', 'open("a")',
                       'x if x else 1', 'sin(x, x)', 'x**x'):
            with self.subTest(unsafe=unsafe), self.assertRaises((ValueError, SyntaxError)):
                safe_expression(unsafe)

    def test_requires_final_marker(self):
        correct, error = grade("2*x", "2*x")
        self.assertFalse(correct)
        self.assertIn("missing Final answer", error)


if __name__ == "__main__":
    unittest.main()
