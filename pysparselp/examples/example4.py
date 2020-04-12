"""Maximum bipartite matching example."""


import numpy as np

from pysparselp.SparseLP import SparseLP


def add_bipartite_constraint(lp, indices):
    columns = indices
    values = np.ones(columns.shape)
    lp.add_linear_constraint_rows(columns, values, lower_bounds=-np.inf, upper_bounds=1)
    columns = indices.T
    values = np.ones(columns.shape)
    lp.add_linear_constraint_rows(columns, values, lower_bounds=-np.inf, upper_bounds=1)


def run():

    n = 50
    np.random.seed(2)
    cost = -np.random.rand(n, n)
    lp = SparseLP()
    indices = lp.add_variables_array(cost.shape, 0, 1, cost)
    add_bipartite_constraint(lp, indices)

    s = lp.solve(method="mehrotra", nb_iter=7, max_time=np.inf)[0]
    print(f"mehrotra final cost:{lp.costsvector.dot(s)}")

    s = lp.solve(method="osqp", nb_iter=1000, max_time=np.inf)[0]
    print(f" osqp final cost:{lp.costsvector.dot(s)}")

    s = lp.solve(
        method="dual_coordinate_ascent", nb_iter=2000, max_time=40, nb_iter_plot=500
    )[0]
    print(f"dual_coordinate_ascent final cost:{lp.costsvector.dot(s)}")

    s = lp.solve(
        method="chambolle_pock_ppd", nb_iter=2000, max_time=10, nb_iter_plot=500
    )[0]
    print(f"chambolle_pock_ppd final cost:{lp.costsvector.dot(s)}")

    x = s[indices]
    print(np.round(x * 1000) / 1000)
    print("done")


if __name__ == "__main__":
    run()
