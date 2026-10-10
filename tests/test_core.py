import math
import unittest

from golden_section_search import golden_section_search


class TestGoldenSectionSearch(unittest.TestCase):
    def test_basic_parabola(self):
        f = lambda x: (x - 2.0) ** 2
        x, fx = golden_section_search(f, 0.0, 5.0)
        self.assertAlmostEqual(x, 2.0, places=6)
        self.assertAlmostEqual(fx, 0.0, places=6)

    def test_offset_parabola_negative_minimum(self):
        f = lambda x: (x + 3.0) ** 2 + 1.0
        x, fx = golden_section_search(f, -10.0, 0.0)
        self.assertAlmostEqual(x, -3.0, places=6)
        self.assertAlmostEqual(fx, 1.0, places=6)

    def test_returns_tuple_of_floats(self):
        f = lambda x: x * x
        result = golden_section_search(f, -1.0, 1.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_tolerance_controls_precision(self):
        f = lambda x: (x - 1.0) ** 2
        x_lo, _ = golden_section_search(f, 0.0, 2.0, tol=1e-2)
        x_hi, _ = golden_section_search(f, 0.0, 2.0, tol=1e-10)
        self.assertLessEqual(abs(x_hi - 1.0), abs(x_lo - 1.0))

    def test_reversed_bounds(self):
        # Caller is free to pass upper < lower; the minimizer's value
        # must still be correct.
        f = lambda x: (x - 2.0) ** 2
        x, fx = golden_section_search(f, 5.0, 0.0)
        self.assertAlmostEqual(fx, 0.0, places=6)
        self.assertAlmostEqual(x, 2.0, places=6)

    def test_minimum_at_left_endpoint(self):
        f = lambda x: x ** 2
        x, fx = golden_section_search(f, 0.0, 2.0)
        # Minimum is at 0; the returned x should be within tol of it
        # and the returned value should be very small.
        self.assertLessEqual(x, 1e-6)
        self.assertLessEqual(fx, 1e-12)

    def test_minimum_at_right_endpoint(self):
        f = lambda x: (x - 2.0) ** 2
        x, fx = golden_section_search(f, 0.0, 2.0)
        self.assertGreaterEqual(x, 2.0 - 1e-6)
        self.assertLessEqual(fx, 1e-12)

    def test_flat_function(self):
        f = lambda x: 3.5
        x, fx = golden_section_search(f, 0.0, 1.0)
        self.assertAlmostEqual(fx, 3.5, places=12)
        self.assertGreaterEqual(x, 0.0)
        self.assertLessEqual(x, 1.0)

    def test_tiny_interval_collapses_to_endpoint(self):
        f = lambda x: (x - 100.0) ** 2
        x, fx = golden_section_search(f, 50.0, 50.0 + 1e-12)
        # Interval is below default tol; routine evaluates once at 50.0
        # and returns that point.
        self.assertEqual(x, 50.0)
        self.assertEqual(fx, 2500.0)

    def test_cosine_minimum_in_bracket(self):
        f = lambda x: math.cos(x)
        # cos(x) on [3, 6] has its minimum at x=pi (~3.14159).
        x, fx = golden_section_search(f, 3.0, 6.0)
        self.assertAlmostEqual(x, math.pi, places=6)
        self.assertAlmostEqual(fx, -1.0, places=6)

    def test_non_symmetric_unimodal(self):
        # exp(x) is monotone increasing; minimum at left endpoint.
        f = lambda x: math.exp(x)
        x, fx = golden_section_search(f, -2.0, 3.0)
        self.assertLessEqual(x, -2.0 + 1e-6)
        self.assertAlmostEqual(fx, math.exp(-2.0), places=6)

    def test_max_iter_limit(self):
        # With max_iter=1 and tol extremely small, the search runs a
        # single shrinking step. We verify the result is still inside
        # the original bracket and the function value is reasonable.
        f = lambda x: (x - 1.0) ** 2
        x, fx = golden_section_search(f, 0.0, 2.0, tol=1e-30, max_iter=1)
        self.assertGreater(x, 0.0)
        self.assertLess(x, 2.0)
        self.assertGreater(fx, -1e-9)

    def test_invalid_tol_raises(self):
        f = lambda x: x * x
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0, tol=0.0)
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0, tol=-1e-5)

    def test_invalid_max_iter_raises(self):
        f = lambda x: x * x
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0, max_iter=0)


if __name__ == "__main__":
    unittest.main()
