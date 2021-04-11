"""Use 10-fold cross-validation to predict test accuracy for each kernel.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

Third-Party Dependencies:

* numpy
* scikit-learn

Runtime Dependencies:

* Requires environment variable named ``CSDS_435_LIBSVM`` containing the path to the LIBSVM library.

"""

import os
import sys

import numpy as np
from sklearn.model_selection import KFold

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


def main() -> int:
    """Use 10-fold cross-validation to predict test accuracy for each kernel."""
    y, x = libsvm_commonutil.svm_read_problem(TRAIN_DATA_PATH)
    ten_fold_cross_validator = KFold(n_splits=10, shuffle=True)

    linear_accuracies = []
    polynomial_accuracies = []

    for train_index, test_index in ten_fold_cross_validator.split(x):
        x_test = np.array(x)[test_index]
        x_train = np.array(x)[train_index]
        y_test = np.array(y)[test_index]
        y_train = np.array(y)[train_index]

        # For reference on options used in training, refer to: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

        trained_linear_kernel = libsvm_svmutil.svm_train(y_train, x_train, "-t 0")
        _, (linear_accuracy, _, _), _ = libsvm_svmutil.svm_predict(y_test, x_test, trained_linear_kernel)
        linear_accuracies.append(linear_accuracy)

        trained_polynomial_kernel = libsvm_svmutil.svm_train(y_train, x_train, "-t 1 -d 5")
        _, (polynomial_accuracy, _, _), _ = libsvm_svmutil.svm_predict(y_test, x_test, trained_polynomial_kernel)
        polynomial_accuracies.append(polynomial_accuracy)

    print(f"Linear Accuracies: {linear_accuracies}")
    print(f"Avg. Linear Accuracy: {sum(linear_accuracies) / len(linear_accuracies)}")

    print(f"Polynomial Accuracies: {polynomial_accuracies}")
    print(f"Avg. Polynomial Accuracy: {sum(polynomial_accuracies) / len(polynomial_accuracies)}")

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
