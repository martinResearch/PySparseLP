"""Comparing different LP solver on a LP problem."""
import matplotlib.pyplot as plt

import numpy as np


def benchmark_methods(
    lp,
    solving_methods,
    nb_iter_plot,
    max_duration,
    display,
    solution=None,
    solution_fig=None,
    gt_solver="scipy_highs-ds",
    display_solution_func=None,
):
    if display:
        solutions_fig = plt.figure()
        fig_curves1 = plt.figure()
        ax_curves1 = fig_curves1.add_subplot(111)
        ax_curves1.set_xlabel("nb of iteration")
        ax_curves1.set_ylabel("distance_to_ground_truth")
        fig_curves2 = plt.figure()
        ax_curves2 = fig_curves2.add_subplot(111)
        ax_curves2.set_xlabel("duration")
        ax_curves2.set_ylabel("distance_to_ground_truth")

    distance_to_ground_truth_curves = {}
    ncol = 5
    nrow = np.ceil((len(solving_methods) + 1) / ncol)

    ground_truth, _ = lp.solve(method=gt_solver)
    assert lp.check_solution(ground_truth)
    if display and display_solution_func is not None:
        ax = solutions_fig.add_subplot(nrow, ncol, 1, title=gt_solver)
        display_solution_func(ax, ground_truth)

    for i, method in enumerate(solving_methods):
        print("\n\n----------------------------------------------------------\n")
        print(f"Solving LP using {method}")

        sol1, elapsed = lp.solve(
            method=method,
            get_timing=True,
            nb_iter=100000,
            max_duration=max_duration,
            ground_truth=ground_truth,
            plot_solution=None,
            nb_iter_plot=nb_iter_plot,
        )
        if display:
            if len(lp.distance_to_ground_truth) > 0:
                ax_curves1.semilogy(
                    lp.itrn_curve, lp.distance_to_ground_truth, label=method
                )
                ax_curves2.semilogy(
                    lp.opttime_curve, lp.distance_to_ground_truth, label=method
                )
            ax_curves1.legend()
            ax_curves2.legend()
            ax = solutions_fig.add_subplot(nrow, ncol, i + 2, title=method)

            if display_solution_func is not None:
                display_solution_func(ax, sol1)

        distance_to_ground_truth_curves[method] = lp.distance_to_ground_truth

    if display:
        plt.show()
    return distance_to_ground_truth_curves
