"""Basis pursuit denoising."""

import numpy as np

from pysparselp.SparseLP import SparseLP

from scipy.stats import laplace


def run(display=True):

    m = 20
    n = 100

    mat = np.random.randn(m, n)
    ratio_zeros = 0.9
    x = np.random.randn(n) * (np.random.rand(n) > ratio_zeros)
    noise = 0.05 * laplace.rvs(size=m)
    y = mat.dot(x) + noise
    lambda_coef = 1.0

    # evalute cost for the x value used to generate the data
    cost_gt = np.sum(np.abs(y - mat.dot(x))) + lambda_coef * np.sum(np.abs(x))
    print(f"cost gt ={cost_gt}")

    lp = SparseLP()
    x_id = lp.add_variables_array((n), lower_bounds=None, upper_bounds=None)
    lp.add_soft_linear_constraint_rows(
        cols=x_id[None, :],
        vals=mat,
        lower_bounds=y,
        upper_bounds=y,
        coef_penalization=1,
    )
    lp.add_soft_linear_constraint_rows(
        cols=x_id[:, None],
        vals=np.ones((n, 1)),
        lower_bounds=0,
        upper_bounds=0,
        coef_penalization=lambda_coef,
    )

    sol, duration = lp.solve("osqp")
    x_opt = sol[x_id]

    cost_opt = np.sum(np.abs(y - mat.dot(x_opt))) + lambda_coef * np.sum(np.abs(x_opt))
    print(f"cost gt ={cost_gt}  cost opt ={cost_opt}")
    assert cost_opt <= cost_gt


if __name__ == "__main__":
    run()
