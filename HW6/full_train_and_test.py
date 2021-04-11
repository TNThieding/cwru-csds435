"""Train each classifier using all the training data and run on test data.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

Runtime Dependencies:

* Requires environment variable named ``CSDS_435_LIBSVM`` containing the path to the LIBSVM library.

"""

import os
import sys

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
    """Train each classifier using all the training data and run on test data."""
    y_train, x_train = libsvm_commonutil.svm_read_problem(TRAIN_DATA_PATH)
    y_test, x_test = libsvm_commonutil.svm_read_problem(TEST_DATA_PATH)

    trained_linear_kernel = libsvm_svmutil.svm_train(y_train, x_train, "-t 0")
    _, (train_linear_accuracy, _, _), _ = libsvm_svmutil.svm_predict(y_train, x_train, trained_linear_kernel)
    _, (test_linear_accuracy, _, _), _ = libsvm_svmutil.svm_predict(y_test, x_test, trained_linear_kernel)

    trained_polynomial_kernel = libsvm_svmutil.svm_train(y_train, x_train, "-t 1 -d 5")
    _, (train_polynomial_accuracy, _, _), _ = libsvm_svmutil.svm_predict(y_train, x_train, trained_polynomial_kernel)
    _, (test_polynomial_accuracy, _, _), _ = libsvm_svmutil.svm_predict(y_test, x_test, trained_polynomial_kernel)

    print(f"Linear Training Accuracy: {train_linear_accuracy}%")
    print(f"Linear Test Accuracy: {test_linear_accuracy}%")
    print(f"Polynomial Training Accuracy: {train_polynomial_accuracy}%")
    print(f"Polynomial Test Accuracy: {test_polynomial_accuracy}%")

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
