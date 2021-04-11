"""Run SVM classifier using linear kernel boosted by AdaBoost.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

Runtime Notes:

* Requires environment variable named ``CSDS_435_LIBSVM`` containing the path to the LIBSVM library.

"""

import math
import os
import random
import sys
from argparse import ArgumentParser

try:
    libsvm_path = os.environ["CSDS_435_LIBSVM"]
except KeyError:
    raise RuntimeError("set environment variable CSDS_435_LIBSVM to point to LIBSVM download")
else:
    sys.path.append(os.path.join(libsvm_path, "python"))

import commonutil as libsvm_commonutil
import svmutil as libsvm_svmutil

EXIT_CODE_SUCCESS = 0
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), "DogsVsCats.train")
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "DogsVsCats.test")


def main() -> int:
    """Run SVM classifier using linear kernel boosted by AdaBoost."""
    argument_parser = ArgumentParser(description=main.__doc__)
    argument_parser.add_argument("--iterations", default=10, type=int, help="number of boosting iterations")
    args = argument_parser.parse_args()

    y_train, x_train = libsvm_commonutil.svm_read_problem(TRAIN_DATA_PATH)

    trained_svms = {}
    weights = {
        0: [1.0 / len(x_train) for _ in x_train]  # set initial weights
    }
    alphas = {}

    for iteration in range(args.iterations):
        # From textbook: "In round i, the tuples from D are sampled to form a training set, Di , of size d. Sampling
        # with replacement is used  -- the same tuple may be selected more than once. Each tuple’s chance of being
        # selected is based on its weight."
        chosen_indicies = random.choices([i for i in range(len(x_train))], weights=weights[iteration], k=len(x_train))

        y_i = [y_train[idx] for idx in chosen_indicies]
        x_i = [x_train[idx] for idx in chosen_indicies]

        trained_linear_kernel = libsvm_svmutil.svm_train(y_i, x_i, "-t 0")
        trained_svms[iteration] = trained_linear_kernel

        predicted_labels, _, _ = libsvm_svmutil.svm_predict(
            y_train, x_train, trained_linear_kernel
        )

        epsilon_t = 0.0
        for index, prediction in enumerate(predicted_labels):
            if prediction != y_train[index]:
                epsilon_t += weights[iteration][index]

        alphas[iteration] = 0.5 * math.log((1 - epsilon_t) / epsilon_t)

        normalization_factor = 0.0
        next_iteration_weights = []
        for index, prediction in enumerate(predicted_labels):
            weight = weights[iteration][index] * math.exp(
                -alphas[iteration] * y_train[index] * prediction
            )

            normalization_factor += weight
            next_iteration_weights.append(weight)

        weights[iteration + 1] = [
            next_iteration_weights[i] / normalization_factor
            for i in range(len(predicted_labels))
        ]

    assert len(trained_svms) == len(alphas)

    # Make prediction using the stored models and their weights.
    y_test, x_test = libsvm_commonutil.svm_read_problem(TEST_DATA_PATH)
    predictions_by_instance = [0 for _ in range(len(y_test))]

    for svm_machine_idx in range(len(trained_svms)):
        predicted_labels, _, _ = libsvm_svmutil.svm_predict(y_test, x_test, trained_svms[svm_machine_idx])
        for label_idx in range(len(predicted_labels)):
            predictions_by_instance[label_idx] += alphas[svm_machine_idx] * predicted_labels[label_idx]

    # Calculate the accuracy of the predictions.
    final_hypotheses = [1.0 if instance >= 0 else -1.0 for instance in predictions_by_instance]
    correct_count = 0

    for index in range(len(final_hypotheses)):
        if final_hypotheses[index] == y_test[index]:
            correct_count += 1

    print(f"AdaBoost Test Accuracy: {correct_count / len(final_hypotheses) * 100}%")

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
