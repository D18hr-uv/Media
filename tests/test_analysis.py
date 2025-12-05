import unittest
import numpy as np
from app import analyze_quarters

class TestAnalysis(unittest.TestCase):
    def test_analyze_quarters(self):
        shap_values = np.array([
            [5, 5, 0.5, 0.5],
            [5, 5, 0.5, 0.5],
            [1, 1, 1.5, 1.5],
            [1, 1, 1.5, 1.5]
        ])
        text, contributions = analyze_quarters(shap_values)

        # Total sum = (5*4) + (0.5*4) + (1*4) + (1.5*4) = 20 + 2 + 4 + 6 = 32
        # Top-Left = 20
        # Percentage = 20/32 * 100 = 62.5%

        self.assertIn("Top-Left area contributed 62.5% to the decision", text)

if __name__ == '__main__':
    unittest.main()
