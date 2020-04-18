"""Test based on the example_l1_svm."""

import json
import os


from pysparselp.examples.example_l1_svm import run

__folder__ = os.path.dirname(__file__)


def test_example_l1_svm(update_results=False):

    percent_valid = run(display=False)

    curves_json_file = os.path.join(__folder__, "test_l1_svm_results.json")
    if update_results:
        with open(curves_json_file, "w") as f:
            json.dump(percent_valid, f)

    with open(curves_json_file, "r") as f:
        percent_valid_expected = json.load(f)

    for k, v1 in percent_valid_expected.items():
        v2 = percent_valid[k]
        assert v1 == v2


if __name__ == "__main__":
    test_example_l1_svm(update_results=False)
