"""Tests using netlib problems."""

import copy
import json
import os


import matplotlib.pyplot as plt

import numpy as np


from pysparselp.SparseLP import SparseLP, solving_methods
from pysparselp.netlib import get_problem

__folder__ = os.path.dirname(__file__)


def solve_netlib(problem_name, display=False, max_time_seconds=30):

    lp_dict = get_problem(problem_name)
    ground_truth = lp_dict["solution"]

    lp = SparseLP()
    nbvar = len(lp_dict["cost_vector"])
    lp.add_variables_array(
        nbvar,
        lower_bounds=lp_dict["lower_bounds"],
        upper_bounds=np.minimum(lp_dict["upper_bounds"], np.max(ground_truth) * 2),
        costs=lp_dict["cost_vector"],
    )
    lp.add_equality_constraints_sparse(lp_dict["a_eq"], lp_dict["b_eq"])
    lp.add_inequality_constraints_sparse(
        lp_dict["a_ineq"], lp_dict["b_lower"], lp_dict["b_upper"]
    )

    print("solving")
    if display:
        f, ax_arr = plt.subplots(3, sharex=True)
        ax_arr[0].set_title("mean absolute distance to solution")
        ax_arr[1].set_title("maximum constraint violation")
        ax_arr[2].set_title("difference with optimum value")

    lp2 = copy.deepcopy(lp)
    lp2.convert_to_one_sided_inequality_system()

    lp = lp2
    assert lp.check_solution(ground_truth)
    cost_gt = lp.costsvector.dot(ground_truth.T)
    print("gt  cost :%f" % cost_gt)

    # testing our methods

    solving_methods2 = solving_methods
    # solving_methods2=['mehrotra']
    distance_to_ground_truth = {}
    for method in solving_methods2:
        print(
            "\n\n----------------------------------------------------------\nSolving LP using %s"
            % method
        )
        sol1, elapsed = lp.solve(
            method=method,
            get_timing=True,
            nb_iter=1000000,
            max_time=max_time_seconds,
            ground_truth=ground_truth,
            ground_truth_indices=np.arange(len(ground_truth)),
            plot_solution=None,
            nb_iter_plot=500,
        )
        distance_to_ground_truth[method] = lp.distance_to_ground_truth
        if display:
            ax_arr[0].semilogy(
                lp.opttime_curve, lp.distance_to_ground_truth, label=method
            )
            ax_arr[1].semilogy(lp.opttime_curve, lp.max_violated_constraint)
            ax_arr[2].semilogy(lp.opttime_curve, lp.pobj_curve - cost_gt)
            ax_arr[0].legend()
            plt.show()
    print("done")
    return distance_to_ground_truth


def trim_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


def test_netlib(
    pb_names=None,
    max_time_seconds: int = 10,
    update_results: bool = False,
    display: bool = False,
):
    if pb_names is None:
        pb_names = ["SC105"]

    for pb_name in pb_names:
        distance_to_ground_truth_curves = solve_netlib(
            pb_name, display=False, max_time_seconds=max_time_seconds
        )

        curves_json_file = os.path.join(__folder__, f"netlib_curves_{pb_name}.json")
        if update_results:
            with open(curves_json_file, "w") as f:
                json.dump(distance_to_ground_truth_curves, f, indent=4)

        with open(curves_json_file, "r") as f:
            distance_to_ground_truth_curves_expected = json.load(f)

        for k, v1 in distance_to_ground_truth_curves_expected.items():
            v2 = distance_to_ground_truth_curves[k]
            tv1, tv2 = trim_length(v1, v2)
            max_diff = np.max(np.abs(np.array(tv1) - np.array(tv2)))
            print(f"{pb_name} max diff {k} = {max_diff}")
            np.testing.assert_almost_equal(*trim_length(v1, v2))


if __name__ == "__main__":

    # test_netlib('afiro')# seems like the solution is not unique
    # test_netlib('SC50B')
    # test_netlib('SC50A')
    # test_netlib('KB2')
    test_netlib(["SC105"], display=False, update_results=False)
    # test_netlib('ADLITTLE')# seems like the solution is not unique
    # test_netlib('SCAGR7')
    # test_netlib('PEROLD')# seems like there is a loading this problem
    # test_netlib('AGG2')
