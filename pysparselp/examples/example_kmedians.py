"""Example using K-Medians

LP formulation inspired from
https://cseweb.ucsd.edu/~dasgupta/291-geom/kmedian.pdf,
http://papers.nips.cc/paper/3478-clustering-via-lp-based-stabilities.pdf
https://www.cs.princeton.edu/courses/archive/fall14/cos521/projects/kmedian.pdf
"""
import matplotlib.pyplot as plt

import numpy as np

from pysparselp.SparseLP import SparseLP


def clustering(points, k, n_center_candidates):

    n = points.shape[0]
    center_candidates = points[np.random.choice(n, n_center_candidates), :]

    pairdistances = np.sqrt(
        np.sum((points[:, None, :] - center_candidates[None, :, :]) ** 2, axis=2)
    )

    lp = SparseLP()
    labeling = lp.add_variables_array(pairdistances.shape, 0, 1, pairdistances)

    used_as_center = lp.add_variables_array(n_center_candidates, 0, 1, 0)
    lp.add_inequality_constraints(
        used_as_center[None, :],
        np.ones((1, n_center_candidates)),
        lower_bounds=0,
        upper_bounds=k,
    )
    lp.add_inequality_constraints(
        labeling, np.ones((n, n_center_candidates)), lower_bounds=1, upper_bounds=1
    )
    # max(labeling,axis=0)<used_as_center
    # the binary variable associated to each  column should be greater than all binary variables on that row
    id_columns = np.ones((n, 1)).dot(used_as_center[None, :])
    columns = np.column_stack((labeling.reshape(-1, 1), id_columns.reshape(-1, 1)))
    values = np.column_stack(
        (np.ones((n * n_center_candidates)), -np.ones((n * n_center_candidates)))
    )
    lp.add_inequality_constraints(columns, values, lower_bounds=None, upper_bounds=0)

    s = lp.solve(method="admm", nb_iter=1000, max_duration=np.inf, nb_iter_plot=500)[0]

    print(lp.costsvector.dot(s))
    x = s[labeling]
    print(np.round(x * 1000) / 1000)

    lp.max_constraint_violation(s)

    label = np.argmax(x, axis=1)
    if not (len(np.unique(label)) == k):
        print("failed")

    cost = 0
    for l in range(n_center_candidates):
        group = np.nonzero(label == l)
        center_id = np.argmin(np.sum(pairdistances[group, :], axis=1))
        cost += np.sum(pairdistances[group, center_id])

    return label, cost


def run(display=False):
    np.random.seed(0)
    k = 5
    n = 500

    prng = np.random.RandomState(0)
    centers = prng.randn(k, 2)
    gt_labels = np.floor(prng.rand(n) * 5).astype(np.int)
    points = 0.4 * prng.randn(n, 2) + centers[gt_labels, :]
    if display:
        plt.ion()
        plt.plot(points[:, 0], points[:, 1], ".")
        plt.draw()
        plt.show()
    n_center_candidates = 50

    label, cost = clustering(points, k, n_center_candidates)
    if display:
        for i in np.arange(n):
            if any(label == i):
                plt.plot(points[label == i, 0], points[label == i, 1], "o")
        plt.draw()
        plt.show()
        plt.axis("equal")
        plt.tight_layout()
        print("done")
    return cost


if __name__ == "__main__":
    run()
