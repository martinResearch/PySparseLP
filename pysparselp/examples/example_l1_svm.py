"""Example using L1-regularized multi-class Support	Vector Machine."""

import matplotlib.pyplot as plt

import numpy as np

from pysparselp.SparseLP import SparseLP, solving_methods


class L1SVM(SparseLP):
    """L1-regularized multi-class Support	Vector Machine  J. Zhu, S. Rosset, T. Hastie, and R. Tibshirani. 1-norm support vector machines. NIPS, 2004."""

    def add_abs_penalization(self, indices, coef_penalization):

        aux = self.add_variables_array(indices.size, upper_bounds=None, lower_bounds=0)

        if np.isscalar(coef_penalization):
            assert coef_penalization > 0
            self.set_costs_variables(aux, np.ones(aux.shape) * coef_penalization)
        # allows a penalization that is different for each edge (could be dependent on an edge detector)
        else:
            assert coef_penalization.shape == aux.shape
            assert np.min(coef_penalization) >= 0
            self.set_costs_variables(aux, np.ones(aux.shape) * coef_penalization)

        # start by adding auxiliary variables

        aux_ravel = aux.ravel()
        indices_ravel = indices.ravel()
        cols = np.column_stack((indices_ravel, aux_ravel))
        vals = np.tile(np.array([1, -1]), [indices.size, 1])
        self.add_inequality_constraints(cols, vals, lower_bounds=None, upper_bounds=0)
        vals = np.tile(np.array([-1, -1]), [indices.size, 1])
        self.add_inequality_constraints(cols, vals, lower_bounds=None, upper_bounds=0)

    def set_data(self, x, classes, nb_classes=None):
        nb_examples = x.shape[0]
        xh = np.hstack((x, np.ones((nb_examples, 1))))
        assert x.shape[0] == len(classes)
        if nb_classes is None:
            nb_classes = np.max(classes) + 1
        nb_features = x.shape[1]

        self.weightsIndices = self.add_variables_array(
            (nb_classes, nb_features + 1), None, None
        )
        self.add_abs_penalization(self.weightsIndices, 1)
        self.epsilonsIndices = self.add_variables_array(
            (nb_examples, 1), upper_bounds=None, lower_bounds=0, costs=1
        )
        e = np.ones((nb_examples, nb_classes))
        e[np.arange(nb_examples), classes] = 0

        # sum(x*weights[classes,:]),axis=1)[:,None]- x.dot(weights)+epsilon>e

        cols1 = self.weightsIndices[classes, :]
        vals1 = xh
        for k in range(nb_classes):
            keep = classes != k
            cols2 = np.tile(self.weightsIndices[[k], :], [nb_examples, 1])
            vals2 = -xh
            vals3 = np.ones(self.epsilonsIndices.shape)
            cols3 = self.epsilonsIndices
            vals = np.column_stack((vals1, vals2, vals3))
            cols = np.column_stack((cols1, cols2, cols3))
            self.add_inequality_constraints(
                cols[keep, :], vals[keep, :], lower_bounds=e[keep, k], upper_bounds=None
            )

    def train(self, method="mehrotra"):

        sol1, elapsed = self.solve(
            method=method,
            get_timing=True,
            nb_iter=2000,
            max_time=np.inf,
            plot_solution=None,
        )
        self.weights = sol1[self.weightsIndices]
        marges = sol1[self.epsilonsIndices]
        self.activeSet = np.nonzero(marges > 1e-3)[0]

    def classify(self, x):
        nb_examples = x.shape[0]
        xh = np.hstack((x, np.ones((nb_examples, 1))))
        scores = xh.dot(self.weights.T)
        classes = np.argmax(scores, axis=1)
        return classes


def run(display=True):

    np.random.seed(1)
    nb_classes = 3
    nb_examples = 1000
    x = np.random.rand(nb_examples, 2)
    xh = np.hstack((x, np.ones((nb_examples, 1))))
    # plt.plot(x[:,0],x[:,1],'.')

    weights = np.random.randn(nb_classes, 2)
    weights = weights / np.sum(weights ** 2, axis=1)[:, None]
    weights = np.hstack((weights, -0.5 * np.sum(weights, axis=1)[:, None]))
    scores = (weights.dot(xh.T)).T
    classes = np.argmax(scores, axis=1)

    colors = ["r", "g", "b"]

    l1svm = L1SVM()
    l1svm.set_data(x, classes)
    percent_valid = {}
    solving_methods_list = list(solving_methods)
    solving_methods_list.remove("mehrotra")  # too slow
    solving_methods_list.remove("scipy_simplex")
    solving_methods_list.remove("scipy_interior_point")
    solving_methods_list.remove("dual_gradient_ascent")  # need to debug
    solving_methods_list.remove("dual_coordinate_ascent")  # need to debug

    for method in solving_methods_list:
        l1svm.train(method=method)
        classes2 = l1svm.classify(x)
        percent_valid[method] = 100 * np.mean(classes == classes2)

    if display:
        colors = ["r", "g", "b"]
        plt.figure()

        for k in range(3):
            plt.plot(x[classes2 == k, 0], x[classes2 == k, 1], ".", color=colors[k])
        plt.plot(
            x[l1svm.activeSet, 0],
            x[l1svm.activeSet, 1],
            "ko",
            markersize=10,
            fillstyle="none",
        )
        plt.axis("equal")

        print("done")
        plt.show()
    return percent_valid


if __name__ == "__main__":
    run()
