"""Test based on the example3."""

import json
import os


from pysparselp.examples.example3 import run

__folder__ = os.path.dirname(__file__)


def test_example3(update_results=False):

    percent_valid = run(display=False)

    curves_json_file = os.path.join(__folder__, "example3_results.json")
    if update_results:
        with open(curves_json_file, "w") as f:
            json.dump(percent_valid, f)

    with open(curves_json_file, "r") as f:
        percent_valid = json.load(f)

    for k, v1 in percent_valid.items():
        v2 = percent_valid[k]
        assert v1 == v2


if __name__ == "__main__":
    test_example3(update_results=False)
