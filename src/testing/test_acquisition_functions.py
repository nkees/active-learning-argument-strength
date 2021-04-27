import unittest
import numpy as np
from src.finetuning.acquisition_functions import Entropy, BALD, VariationRatios


class TestAcquisitionFunctions(unittest.TestCase):

    test_softmax_outputs = np.array(
        [
            [
                [0.23581544, 0.76418456],
                [0.3127814, 0.6872186],
                [0.32980248, 0.67019752],
                [0.19946607, 0.80053393],
                [0.2833897, 0.7166103],
            ],
            [
                [0.31011493, 0.68988507],
                [0.24733438, 0.75266562],
                [0.28629061, 0.71370939],
                [0.2771737, 0.7228263],
                [0.28372344, 0.71627656],
            ],
        ]
    )

    def test_entropy(self, test_softmax_outputs=test_softmax_outputs):
        acquisition_function = Entropy
        uncertainties = acquisition_function.calculate_uncertainty(test_softmax_outputs)
        self.assertSequenceEqual(
            np.around(uncertainties, 6).tolist(),
            np.around(
                np.array([0.84568739, 0.85552968, 0.89090565, 0.79223513, 0.86025174]),
                6,
            ).tolist(),
        )

    def test_variation_ratios(self, test_softmax_outputs=test_softmax_outputs):
        acquisition_function = VariationRatios
        uncertainties = acquisition_function.calculate_uncertainty(test_softmax_outputs)
        self.assertSequenceEqual(
            np.around(uncertainties, 6).tolist(), np.around(np.array([0.6, 0.6, 0.6, 0.6, 0.6]), 6).tolist()
        )

    def test_bald(self, test_softmax_outputs=test_softmax_outputs):
        acquisition_function = BALD
        uncertainties = acquisition_function.calculate_uncertainty(test_softmax_outputs)
        self.assertSequenceEqual(
            np.around(uncertainties, 6).tolist(),
            np.around(
                np.array(
                    [
                        5.02835800e-03,
                        3.83774003e-03,
                        1.60279918e-03,
                        6.02008234e-03,
                        9.88734015e-08,
                    ]
                ),
                6,
            ).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
