"""Optional, narrow external commission primitive, not engine acceptance."""

import unittest


class ProbeTest(unittest.TestCase):
    def test_calculations_and_default_scaling_are_separated(self):
        from experiments.backtrader_commission_probe import assess

        # Adapter supplies actual external library observations in the real run.
        result = assess("0.02", "0.0002", "0.02")
        self.assertTrue(result["explicit_units_passed"])
        self.assertTrue(result["default_scale_mismatch_detected"])

    def test_large_or_nonfinite_error_does_not_pass(self):
        from experiments.backtrader_commission_probe import assess

        for observed in ("nan", "inf", "0.021"):
            with self.subTest(observed=observed):
                self.assertFalse(assess(observed, "0.0002", "0.02")["explicit_units_passed"])
