"""Maximum bipartite matching example."""


import numpy as np

from pysparselp.SparseLP import SparseLP


def add_bipartite_constraint(LP, indices):
    cols = indices
    vals = np.ones(cols.shape)
    LP.add_linear_constraint_rows(cols, vals, lowerbounds=-np.inf, upperbounds=1)
    cols = indices.T
    vals = np.ones(cols.shape)
    LP.add_linear_constraint_rows(cols, vals, lowerbounds=-np.inf, upperbounds=1)


def run():

    n = 50
    np.random.seed(2)
    Cost = -np.random.rand(n, n)
    LP = SparseLP()
    indices = LP.add_variables_array(Cost.shape, 0, 1, Cost)
    add_bipartite_constraint(LP, indices)

    s = LP.solve(method="Mehrotra", nb_iter=7, max_time=np.inf)[0]
    print(LP.costsvector.dot(s))

    s = LP.solve(
        method="dual_coordinate_ascent", nb_iter=2000, max_time=40, nb_iter_plot=500
    )[0]
    print(LP.costsvector.dot(s))

    s = LP.solve(
        method="chambolle_pock_ppd", nb_iter=2000, max_time=10, nb_iter_plot=500
    )[0]
    print(LP.costsvector.dot(s))

    x = s[indices]
    print(np.round(x * 1000) / 1000)
    print("done")


if __name__ == "__main__":
    run()
