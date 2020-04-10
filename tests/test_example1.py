
"""Test based on the example1 (pott image model)."""

import json
import os

import numpy as np
import numpy.testing

from pysparselp.examples.example1 import run


__folder__ = os.path.dirname(__file__)


def trim_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


def test_example1(update_results=False):

    distance_to_ground_truth_curves = run(display=False)

    curves_json_file = os.path.join(__folder__, "example1_curves.json")
    if update_results:
        with open(curves_json_file, "w") as f:
            json.dump(distance_to_ground_truth_curves, f, indent=4)

    with open(curves_json_file, "r") as f:
        distance_to_ground_truth_curves_expected = json.load(f)

    for k, v1 in distance_to_ground_truth_curves.items():
        v2 = distance_to_ground_truth_curves_expected[k]
        tv1, tv2 = trim_length(v1, v2)
        max_diff = np.max(np.abs(np.array(tv1) - np.array(tv2)))
        print(f"max diff {k} = {max_diff}")
        numpy.testing.assert_almost_equal(*trim_length(v1, v2))


if __name__ == "__main__":
    test_example1(update_results=False)
