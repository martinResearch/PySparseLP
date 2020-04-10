"""Example using L1-regularized multi-class Support	Vector Machine."""

import matplotlib.pyplot as plt

import numpy as np

from pysparselp.SparseLP import SparseLP, solving_methods


class L1SVM(SparseLP):
    """L1-regularized multi-class Support	Vector Machine  J. Zhu, S. Rosset, T. Hastie, and R. Tibshirani. 1-norm support vector machines. NIPS, 2004."""

    def add_abs_penalization(self, indices, coefpenalization):

        aux = self.add_variables_array(indices.size, upperbounds=None, lowerbounds=0)

        if np.isscalar(coefpenalization):
            assert coefpenalization > 0
            self.set_costs_variables(aux, np.ones(aux.shape) * coefpenalization)
        # allows a penalization that is different for each edge (could be dependent on an edge detector)
        else:
            assert coefpenalization.shape == aux.shape
            assert np.min(coefpenalization) >= 0
            self.set_costs_variables(aux, np.ones(aux.shape) * coefpenalization)

        # start by adding auxilary variables

        aux_ravel = aux.ravel()
        indices_ravel = indices.ravel()
        cols = np.column_stack((indices_ravel, aux_ravel))
        vals = np.tile(np.array([1, -1]), [indices.size, 1])
        self.add_linear_constraint_rows(cols, vals, lowerbounds=None, upperbounds=0)
        vals = np.tile(np.array([-1, -1]), [indices.size, 1])
        self.add_linear_constraint_rows(cols, vals, lowerbounds=None, upperbounds=0)

    def set_data(self, x, classes, nbClasses=None):
        nbExamples = x.shape[0]
        xh = np.hstack((x, np.ones((nbExamples, 1))))
        assert x.shape[0] == len(classes)
        if nbClasses is None:
            nbClasses = np.max(classes) + 1
        nbFeatures = x.shape[1]

        self.weightsIndices = self.add_variables_array(
            (nbClasses, nbFeatures + 1), None, None
        )
        self.add_abs_penalization(self.weightsIndices, 1)
        self.epsilonsIndices = self.add_variables_array(
            (nbExamples, 1), upperbounds=None, lowerbounds=0, costs=1
        )
        e = np.ones((nbExamples, nbClasses))
        e[np.arange(nbExamples), classes] = 0

        # sum(x*weights[classes,:]),axis=1)[:,None]- x.dot(weights)+epsilon>e

        cols1 = self.weightsIndices[classes, :]
        vals1 = xh
        for k in range(nbClasses):
            keep = classes != k
            cols2 = np.tile(self.weightsIndices[[k], :], [nbExamples, 1])
            vals2 = -xh
            vals3 = np.ones(self.epsilonsIndices.shape)
            cols3 = self.epsilonsIndices
            vals = np.column_stack((vals1, vals2, vals3))
            cols = np.column_stack((cols1, cols2, cols3))
            self.add_linear_constraint_rows(
                cols[keep, :], vals[keep, :], lowerbounds=e[keep, k], upperbounds=None
            )

    def train(self, method="Mehrotra"):

        sol1, elapsed = self.solve(
            method=method,
            getTiming=True,
            nb_iter=2000,
            max_time=np.inf,
            plot_solution=None,
        )
        self.weights = sol1[self.weightsIndices]
        marges = sol1[self.epsilonsIndices]
        self.activeSet = np.nonzero(marges > 1e-3)[0]

    def classify(self, x):
        nbExamples = x.shape[0]
        xh = np.hstack((x, np.ones((nbExamples, 1))))
        scores = xh.dot(self.weights.T)
        classes = np.argmax(scores, axis=1)
        return classes


def run(display=True):

    np.random.seed(1)
    nbClasses = 3
    nbExamples = 1000
    x = np.random.rand(nbExamples, 2)
    xh = np.hstack((x, np.ones((nbExamples, 1))))
    # plt.plot(x[:,0],x[:,1],'.')

    weights = np.random.randn(nbClasses, 2)
    weights = weights / np.sum(weights ** 2, axis=1)[:, None]
    weights = np.hstack((weights, -0.5 * np.sum(weights, axis=1)[:, None]))
    scores = (weights.dot(xh.T)).T
    classes = np.argmax(scores, axis=1)

    colors = ["r", "g", "b"]

    l1svm = L1SVM()
    l1svm.set_data(x, classes)
    percent_valid = {}

    solving_methods.remove("Mehrotra")  # too slow
    solving_methods.remove("ScipyLinProg")
    solving_methods.remove("dual_gradient_ascent")  # need to debug
    solving_methods.remove("dual_coordinate_ascent")  # need to debug

    for method in solving_methods:
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
