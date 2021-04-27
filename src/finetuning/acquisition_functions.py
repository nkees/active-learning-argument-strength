import numpy as np


def predictive_entropy(softmax_outputs):
    probs = softmax_outputs.mean(axis=0)
    return -np.sum(probs * np.log2(probs), axis=1)


class Entropy:
    @staticmethod
    def calculate_uncertainty(softmax_outputs):
        return predictive_entropy(softmax_outputs)


class BALD:
    @staticmethod
    def calculate_uncertainty(softmax_outputs):
        mean_entropy = predictive_entropy(softmax_outputs)

        entropy_per_pair = -np.sum(softmax_outputs * np.log2(softmax_outputs), axis=2)
        entropy_mean = np.mean(entropy_per_pair, axis=0)

        return mean_entropy - entropy_mean


class VariationRatios:
    @staticmethod
    def calculate_uncertainty(softmax_outputs):
        predicted = softmax_outputs.argmax(axis=2)

        num_1 = (predicted == 1).sum(axis=0)
        num_0 = (predicted == 0).sum(axis=0)
        variations = np.array([num_0, num_1])

        ratios = variations / len(variations[0])
        variation_ratios = 1 - np.max(ratios, axis=0)
        return variation_ratios


acquisition_functions = {
    "entropy": Entropy,
    "bald": BALD,
    "variation_ratios": VariationRatios,
}
