import unittest
import numpy as np
import os

print(os.getcwd())
from src.finetuning.active_am import convert_logits_to_softmax


class TestActiveArgumentMining(unittest.TestCase):
    test_logits = np.array(
        [
            [
                [-0.82011193, 0.35564795],
                [-0.6273028, 0.15984508],
                [-0.45624253, 0.25283602],
                [-0.677311, 0.7123238],
                [-0.5869793, 0.3407299],
            ],
            [
                [-0.6542956, 0.14528647],
                [-0.5532149, 0.559665],
                [-0.4459999, 0.46746853],
                [-0.57116693, 0.38735765],
                [-0.67816734, 0.24789906],
            ],
        ]
    )

    def test_convert_logits_to_softmax(self, test_logits=test_logits):
        test_softmax_outputs = np.around(convert_logits_to_softmax(test_logits), 6)
        self.assertSequenceEqual(
            test_softmax_outputs.tolist(),
            np.around(
                np.array(
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
                ),
                6,
            ).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
