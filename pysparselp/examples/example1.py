"""Example using a pott image model that can be exactly solved with graphcut."""

import matplotlib.pyplot as plt

import maxflow  # pip install PyMaxflow

import numpy as np

from pysparselp.SparseLP import SparseLP, solving_methods


class ImageLP(SparseLP):
    """Specialization of the generic SparseLP to define linear relaxation of pott image models."""

    def add_penalized_differences(self, I, J, coefpenalization):
        assert I.size == J.size
        maxDiff = np.maximum(
            self.upperbounds[I] - self.lowerbounds[J],
            self.upperbounds[J] - self.lowerbounds[I],
        )
        aux = self.add_variables_array(
            I.shape, upperbounds=maxDiff, lowerbounds=0, costs=coefpenalization
        )
        if np.isscalar(coefpenalization):
            assert coefpenalization > 0
        # allows a penalization that is different for each edge (could be dependent on an edge detector)
        else:
            assert coefpenalization.shape == aux.shape
            assert np.min(coefpenalization) >= 0
        aux_ravel = aux.ravel()
        I_ravel = I.ravel()
        J_ravel = J.ravel()
        cols = np.column_stack((I_ravel, J_ravel, aux_ravel))
        vals = np.tile(np.array([1, -1, -1]), [I.size, 1])
        self.add_linear_constraint_rows(cols, vals, lowerbounds=None, upperbounds=0)
        vals = np.tile(np.array([-1, 1, -1]), [I.size, 1])
        self.add_linear_constraint_rows(cols, vals, lowerbounds=None, upperbounds=0)

    def add_pott_horizontal(self, indices, coefpenalization):
        self.add_penalized_differences(indices[:, 1:], indices[:, :-1], coefpenalization)

    def add_pott_vertical(self, indices, coefpenalization):
        self.add_penalized_differences(indices[1:, :], indices[:-1, :], coefpenalization)

    def add_pott_model(self, indices, coefpenalization):
        self.add_pott_horizontal(indices, coefpenalization)
        self.add_pott_vertical(indices, coefpenalization)


def build_linear_program(imageSize, coefPotts, coefMul):
    nLabels = 1
    np.random.seed(1)

    size_image = (imageSize, imageSize, nLabels)

    # we multiply all term by theis constant because the graph cut algorithm take integer weights.

    unary_terms = np.round(
        coefMul
        * ((np.random.rand(size_image[0], size_image[1], size_image[2])) * 2 - 1)
    )
    coefPotts = round(coefPotts * coefMul)

    g = maxflow.Graph[int](0, 0)
    nodeids = g.add_grid_nodes(unary_terms.shape)

    alpha = coefPotts
    g.add_grid_edges(nodeids, alpha)
    # Add the terminal edges.
    g.add_grid_tedges(nodeids, unary_terms * 0, unary_terms)

    print("calling maxflow")
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    img2 = np.int_(np.logical_not(sgm))
    plt.imshow(img2[:, :, 0], cmap=plt.cm.gray, interpolation="nearest")

    LP = ImageLP()

    indices = LP.add_variables_array(
        shape=size_image, lowerbounds=0, upperbounds=1, costs=unary_terms / coefMul
    )

    groundTruth = img2
    groundTruthIndices = indices

    LP.add_pott_model(indices, coefpenalization=coefPotts / coefMul)
    return LP, groundTruth, groundTruthIndices, unary_terms


def run(display=True):

    imageSize = 50
    coefMul = 500
    coefPotts = 0.5
    LP, groundTruth, groundTruthIndices, unary_terms = build_linear_program(
        imageSize, coefPotts, coefMul
    )

    print("solving")

    if display:
        im = plt.imshow(
            unary_terms[:, :, 0] / coefMul,
            cmap=plt.cm.Greys_r,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        ax_curves1 = plt.gca()
        ax_curves1.set_xlabel("nb of iteration")
        ax_curves1.set_ylabel("distanceToGroundTruth")
        ax_curves2 = plt.gca()
        ax_curves2.set_xlabel("duration")
        ax_curves2.set_ylabel("distanceToGroundTruth")

    def plot_solution(niter, solution, is_active_variable=None):
        image = solution[groundTruthIndices]
        # imwrite('ter%05d.png'%niter,solution[indices][:,:,0])
        # imwrite('diff_iter%05d.png'%niter,np.diff(solution[indices][:,:,0]))
        im.set_array(image[:, :, 0])
        # im.set_array(np.diff(image[:,:,0]))
        plt.draw()

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(2, 5, 1, title="graph cut")
        ax.imshow(groundTruth[:, :, 0], cmap=plt.cm.Greys_r, interpolation="none")
        ax.axis("off")

    # simplex much too slow for images larger than 20 by 20
    # LP2=copy.deepcopy(LP)
    # LP2.convert_to_one_sided_inequality_system()
    # sol1,elapsed=LP2.solve(method='ScipyLinProg',force_integer=False,getTiming=True,nb_iter=100,max_time=10,groundTruth=groundTruth,groundTruthIndices=indices,plot_solution=None)

    solving_methods2 = [
        m for m in solving_methods if (m not in ["scipy_linprog"])
    ]  # remove scipy_linprog because it is too slow

    distanceToGroundTruthCurves = {}

    for i, method in enumerate(solving_methods2):
        print(
            "\n\n----------------------------------------------------------\nSolving LP using %s"
            % method
        )

        sol1, elapsed = LP.solve(
            method=method,
            getTiming=True,
            nb_iter=1000000,
            max_time=15,
            groundTruth=groundTruth,
            groundTruthIndices=groundTruthIndices,
            plot_solution=None,
            nb_iter_plot=500,
        )
        if display:
            ax_curves1.semilogy(LP.itrn_curve, LP.distanceToGroundTruth, label=method)
            ax_curves2.semilogy(
                LP.opttime_curve, LP.distanceToGroundTruth, label=method
            )
            ax_curves1.legend()
            ax_curves2.legend()
            ax = fig.add_subplot(2, 5, i + 2, title=method)
            ax.imshow(
                sol1[groundTruthIndices][:, :, 0],
                cmap=plt.cm.Greys_r,
                interpolation="none",
                vmin=0,
                vmax=1,
            )
            ax.axis("off")
            plt.draw()

        distanceToGroundTruthCurves[method] = LP.distanceToGroundTruth

    if display:
        plt.tight_layout()
        # plt.figure()
        # plt.plot(LP.itrn_curve,LP.dopttime_curve,'g',label='ADMM')
        # plt.draw()
        # plt.show()
        print("done")
        plt.show()

    return distanceToGroundTruthCurves


if __name__ == "__main__":
    run()
