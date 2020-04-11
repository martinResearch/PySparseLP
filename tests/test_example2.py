"""Test based on the example2."""

import os

import numpy as np
import numpy.testing

from pysparselp.examples.example2 import run


__folder__ = os.path.dirname(__file__)


def test_example2(update_results=False):

    sum_abs_diff, nb_zeros_lp = run(display=False)

    assert sum_abs_diff < 14.02
    print(nb_zeros_lp)
    assert nb_zeros_lp == 236


if __name__ == "__main__":
    test_example2()
