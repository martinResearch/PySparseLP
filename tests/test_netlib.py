"""Tests using netlib problems."""

import copy
import json
import os


import matplotlib.pyplot as plt

import numpy as np


from pysparselp.SparseLP import SparseLP, solving_methods
from pysparselp.netlib import get_problem

__folder__ = os.path.dirname(__file__)


def solve_netlib(problem_name, display=False, max_duration_seconds=30):

    lp_dict = get_problem(problem_name)
    ground_truth = lp_dict["solution"]

    lp = SparseLP()
    nbvar = len(lp_dict["cost_vector"])

    lp.add_variables_array(
        nbvar,
        lower_bounds=lp_dict["lower_bounds"],
        upper_bounds=lp_dict["upper_bounds"],
        costs=lp_dict["cost_vector"],
    )
    lp.add_equality_constraints_sparse(lp_dict["a_eq"], lp_dict["b_eq"])
    lp.add_inequality_constraints_sparse(
        lp_dict["a_ineq"], lp_dict["b_lower"], lp_dict["b_upper"]
    )

    print("solving")

    if display:
        fig, ax_arr = plt.subplots(3, sharex=True)
        fig.set_size_inches(15, 10)
        fig.suptitle(problem_name)
        ax_arr[0].set_title("mean absolute distance to solution")
        ax_arr[1].set_title("maximum constraint violation")
        ax_arr[2].set_title("absolute difference with optimum value")

    lp2 = copy.deepcopy(lp)
    lp2.convert_to_one_sided_inequality_system()

    lp = lp2
    assert lp.check_solution(ground_truth)
    cost_gt = lp.costsvector.dot(ground_truth.T)
    print("gt  cost :%f" % cost_gt)

    # testing our methods

    solving_methods2 = solving_methods[:6]
    all_method_options_list = {}
    all_method_options_list["chambolle_pock_linesearch"] = {
        method: {"method": method}
        for method in ("standard", "activation", "xyseparate", "without_linesearch")
    }

    # solving_methods2=['mehrotra']
    distance_to_ground_truth = {}
    for method in solving_methods2:

        if method in all_method_options_list:
            method_options_list = all_method_options_list[method]
        else:
            method_options_list = {"default": None}

        for options_name, method_options in method_options_list.items():

            if method in all_method_options_list:
                full_name = f"{method} {options_name}"
            else:
                full_name = method

            print(
                f"\n\n----------------------------------------------------------\nSolving LP using {full_name}"
            )

            sol1, elapsed = lp.solve(
                method=method,
                get_timing=True,
                nb_iter=1000000,
                max_duration=max_duration_seconds,
                ground_truth=ground_truth,
                ground_truth_indices=np.arange(len(ground_truth)),
                plot_solution=None,
                nb_iter_plot=500,
                method_options=method_options,
            )
            distance_to_ground_truth[full_name] = lp.distance_to_ground_truth
            if display:
                ax_arr[0].semilogy(
                    lp.opttime_curve, lp.distance_to_ground_truth, label=full_name
                )
                ax_arr[0].set_xlabel("duration in seconds")
                ax_arr[0].set_xlim([0, max_duration_seconds * 2.1])
                ax_arr[1].semilogy(lp.opttime_curve, lp.max_violated_constraint)
                ax_arr[1].set_xlabel("duration in seconds")
                ax_arr[2].semilogy(lp.opttime_curve, np.abs(lp.pobj_curve - cost_gt))
                ax_arr[2].set_xlabel("duration in seconds")
                ax_arr[0].legend(loc='upper right')

    fig.savefig(os.path.join(__folder__, f"curves_{problem_name}.svg"))
    print("done")
    return distance_to_ground_truth


def trim_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


def generic_test_netlib(
    pb_names=None,
    max_duration_seconds: int = 15,
    update_results: bool = False,
    display: bool = False,
):
    if isinstance(pb_names, str):
        raise BaseException("you should provide a list of strings")

    if pb_names is None:
        pb_names = ["SC50B"]

    for pb_name in pb_names:
        distance_to_ground_truth_curves = solve_netlib(
            pb_name, display=display, max_duration_seconds=max_duration_seconds
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
            if np.all(np.isnan(v1)):
                assert(np.all(np.isnan(v2)))
            max_diff = np.max(np.abs(np.array(tv1) - np.array(tv2)))
            print(f"{pb_name} max diff {k} = {max_diff}")
            np.testing.assert_almost_equal(*trim_length(v1, v2))


def test_SC50B():
    generic_test_netlib(['SC50B'], update_results=False, display=True)


if __name__ == "__main__":

    # problems for which mehrotra find the same solution as the provided one
    problems = ['SC50B', 'SC50A', 'KB2', "SC105", 'SCAGR7']
    problems = ['SC50B']
    # problem that seems to have several solutions (mehrotra find a different solution from the provided one)
    # badly conditioned ?
    # "AFIRO" 'ADLITTLE'

    # problem for which the provide solution seems to violate constraints
    # 'PEROLD','AGG2'

    generic_test_netlib(problems, update_results=True, display=True)
