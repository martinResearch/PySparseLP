"""Module to generate random LP problems."""

import copy

import matplotlib.pyplot as plt

import numpy as np

import scipy.sparse

from . import SparseLP, solving_methods


def rand_sparse(shape, sparsity):
    if isinstance(shape, tuple) or isinstance(shape, list):
        return (
            np.round(np.random.randn(*shape) * 100)
            * (np.random.rand(*shape) < sparsity)
            / 100
        )
    else:
        return (
            np.round(np.random.randn(shape) * 100)
            * (np.random.rand(shape) < sparsity)
            / 100
        )


def generate_random_lp(nbvar, n_eq, n_ineq, sparsity):

    # maybe could have a look at https://www.jstor.org/stable/3689906?seq=1#page_scan_tab_contents
    # https://deepblue.lib.umich.edu/bitstream/handle/2027.42/3549/bam8969.0001.001.pdf
    feasible_x = rand_sparse(nbvar, sparsity=1)

    if n_ineq > 0:
        while True:  # make sure the matrix is not empty
            a_ineq = scipy.sparse.csr_matrix(rand_sparse((n_ineq, nbvar), sparsity))
            keep = (
                (a_ineq != 0).dot(np.ones(nbvar))
            ) >= 2  # keep only rows with at least two non zeros values
            if np.sum(keep) >= 1:
                break
        bmin = a_ineq.dot(feasible_x)
        b_upper = (
            np.ceil((bmin + abs(rand_sparse(n_ineq, sparsity))) * 1000) / 1000
        )  # make v feasible
        b_lower = None  # bmin-abs(rand_sparse(n_ineq,sparsity))
        a_ineq = a_ineq[keep, :]
        b_upper = b_upper[keep]

    costs = rand_sparse(nbvar, sparsity=1)

    t = rand_sparse(nbvar, sparsity=1)
    lower_bounds = feasible_x + np.minimum(0, t)
    upper_bounds = feasible_x + np.maximum(0, t)

    lp = SparseLP()
    lp.add_variables_array(
        nbvar, lower_bounds=lower_bounds, upper_bounds=upper_bounds, costs=costs
    )
    if n_eq > 0:
        a_eq = scipy.sparse.csr_matrix(rand_sparse((n_eq, nbvar), sparsity))
        b_eq = a_eq.dot(feasible_x)
        keep = (
            (a_eq != 0).dot(np.ones(nbvar))
        ) >= 2  # keep only rows with at least two non zeros values
        a_eq = a_eq[keep, :]
        b_eq = b_eq[keep]
        if a_eq.indices.size > 0:
            lp.add_equality_constraints_sparse(a_eq, b_eq)
    if n_ineq > 0 and a_ineq.indices.size > 0:
        lp.add_inequality_constraints_sparse(a_ineq, b_lower, b_upper)

    assert lp.check_solution(feasible_x)
    return lp, feasible_x


if __name__ == "__main__":
    plt.ion()

    lp, v = generate_random_lp(nbvar=30, n_eq=1, n_ineq=30, sparsity=0.2)
    lp2 = copy.deepcopy(lp)
    lp2.convert_to_one_sided_inequality_system()
    scipy_sol, elapsed = lp2.solve(
        method="scipy_linprog", force_integer=False, get_timing=True, nb_iter=100000
    )
    cost_scipy = scipy_sol.dot(lp2.costsvector.T)
    maxv = lp2.max_constraint_violation(scipy_sol)
    if maxv > 1e-8:
        print("not expected")
        raise

    ground_truth = scipy_sol
    solving_methods2 = list(set(solving_methods) - set(["dual_gradient_ascent"]))

    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].set_title("mean absolute distance to solution")
    axarr[1].set_title("maximum constraint violation")
    axarr[2].set_title("difference with optimum value")
    max_duration = 2
    for method in solving_methods2:
        sol1, elapsed = lp2.solve(
            method=method, max_duration=max_duration, ground_truth=ground_truth
        )
        axarr[0].semilogy(
            lp2.opttime_curve,
            np.maximum(lp2.distance_to_ground_truth, 1e-18),
            label=method,
        )
        axarr[1].semilogy(
            lp2.opttime_curve, np.maximum(lp2.max_violated_constraint, 1e-18)
        )
        axarr[2].semilogy(
            lp2.opttime_curve, np.maximum(lp2.pobj_curve - cost_scipy, 1e-18)
        )
        axarr[0].legend()
        plt.show()
    print("done")
