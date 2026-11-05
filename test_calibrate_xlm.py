import unittest

import numpy as np

from calibrate_xlm import score, tune


class CalibrationTests(unittest.TestCase):
    def test_perfect_predictions_keep_unit_weights(self):
        labels = np.tile(np.arange(4), 5)
        probabilities = np.tile(np.eye(4) * 0.85 + 0.0375, (5, 1))
        weights = tune(labels, probabilities)
        np.testing.assert_array_equal(weights, np.ones(4))
        self.assertEqual(score(labels, (probabilities * weights).argmax(axis=1)), 1.0)


if __name__ == "__main__":
    unittest.main()