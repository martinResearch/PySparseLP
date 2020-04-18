"""Test based on the example_sparse_inv_covariance"""

import os

from pysparselp.examples.example_sparse_inv_covariance import run


__folder__ = os.path.dirname(__file__)


def test_sparse_inv_covariance(update_results=False):

    sum_abs_diff, nb_zeros_lp = run(display=False)

    assert sum_abs_diff < 14.02
    print(nb_zeros_lp)
    assert nb_zeros_lp >= 232


if __name__ == "__main__":
    test_sparse_inv_covariance()
