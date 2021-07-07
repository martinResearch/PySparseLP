"""Module to generate random LP problems."""


import numpy as np

from pysparselp.SparseLP import SparseLP, solving_methods
from pysparselp.tools import compare_methods

import scipy.sparse


def rand_sparse(shape, sparsity):
    if isinstance(shape, tuple) or isinstance(shape, list):
        return (
            np.round(np.random.randn(*shape) * 100)
            * (np.random.rand(*shape) < (1 - sparsity))
            / 100
        )
    else:
        return (
            np.round(np.random.randn(shape) * 100)
            * (np.random.rand(shape) < (1 - sparsity))
            / 100
        )


def generate_random_lp(nbvar, n_eq, n_ineq, sparsity, tol=1e-10, seed=None):

    # maybe could have a look at https://www.jstor.org/stable/3689906?seq=1#page_scan_tab_contents
    # https://deepblue.lib.umich.edu/bitstream/handle/2027.42/3549/bam8969.0001.001.pdf
    scipy_succeed = False
    if seed is not None:
        np.random.seed(seed)

    while not scipy_succeed:
        feasible_x = rand_sparse(nbvar, sparsity=0)

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

        costs = rand_sparse(nbvar, sparsity=0)

        t = rand_sparse(nbvar, sparsity=0)
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
        try:
            scipy_sol_interior_point, elapsed = lp.solve(
                method="scipy_highs-ds",
                get_timing=True,
                nb_iter=100000,
                error_if_fail=True,
            )
            scipy_sol_revised_simplex, elapsed = lp.solve(
                method="scipy_highs-ipm",
                get_timing=True,
                nb_iter=100000,
                error_if_fail=True,
            )
            max_diff = np.max(abs(scipy_sol_revised_simplex - scipy_sol_interior_point))
            if max_diff < 1e-8:
                scipy_succeed = True
            else:
                print(
                    f"interior point and simplex solution differ too much (max diff= {max_diff}), generating new problem"
                )
        except BaseException:
            scipy_succeed = False
            print("could not solve with scipy, generating new problem")

    return lp, feasible_x, scipy_sol_revised_simplex


def _main(display=True):

    lp, v, scipy_sol = generate_random_lp(
        nbvar=10, n_eq=1, n_ineq=10, sparsity=0.8, seed=0
    )

    solving_methods2 = list(solving_methods)
    solving_methods2.remove("dual_gradient_ascent")

    compare_methods(
        lp, solving_methods2, ground_truth=scipy_sol, display=True, max_duration=10
    )


if __name__ == "__main__":
    _main()
