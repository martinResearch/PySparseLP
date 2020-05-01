"""Example using a pott image model that can be exactly solved with graphcut."""

import matplotlib.pyplot as plt

import maxflow  # pip install PyMaxflow

import numpy as np

from pysparselp.SparseLP import SparseLP, solving_methods


class ImageLP(SparseLP):
    """Specialization of the generic SparseLP to define linear relaxation of pott image models."""

    def add_penalized_differences(self, ids1, ids2, coef_penalization):
        assert ids1.size == ids2.size
        max_diff = np.maximum(
            self.upper_bounds[ids1] - self.lower_bounds[ids2],
            self.upper_bounds[ids2] - self.lower_bounds[ids1],
        )
        aux = self.add_variables_array(
            ids1.shape, upper_bounds=max_diff, lower_bounds=0, costs=coef_penalization
        )
        if np.isscalar(coef_penalization):
            assert coef_penalization > 0
        # allows a penalization that is different for each edge (could be dependent on an edge detector)
        else:
            assert coef_penalization.shape == aux.shape
            assert np.min(coef_penalization) >= 0
        aux_ravel = aux.ravel()
        row_ravel = ids1.ravel()
        col_ravel = ids2.ravel()
        cols = np.column_stack((row_ravel, col_ravel, aux_ravel))
        vals = np.tile(np.array([1, -1, -1]), [ids1.size, 1])
        self.add_inequality_constraints(cols, vals, lower_bounds=None, upper_bounds=0)
        vals = np.tile(np.array([-1, 1, -1]), [ids1.size, 1])
        self.add_inequality_constraints(cols, vals, lower_bounds=None, upper_bounds=0)

    def add_pott_horizontal(self, indices, coef_penalization):
        self.add_penalized_differences(
            indices[:, 1:], indices[:, :-1], coef_penalization
        )

    def add_pott_vertical(self, indices, coef_penalization):
        self.add_penalized_differences(
            indices[1:, :], indices[:-1, :], coef_penalization
        )

    def add_pott_model(self, indices, coef_penalization):
        self.add_pott_horizontal(indices, coef_penalization)
        self.add_pott_vertical(indices, coef_penalization)


def build_linear_program(image_size, coef_potts, coef_mul):
    nb_labels = 1
    np.random.seed(1)

    size_image = (image_size, image_size, nb_labels)

    # we multiply all term by this constant because the graph cut algorithm take integer weights.

    unary_terms = np.round(
        coef_mul
        * ((np.random.rand(size_image[0], size_image[1], size_image[2])) * 2 - 1)
    )
    coef_potts = round(coef_potts * coef_mul)

    g = maxflow.Graph[int](0, 0)
    nodeids = g.add_grid_nodes(unary_terms.shape)

    alpha = coef_potts
    g.add_grid_edges(nodeids, alpha)
    # Add the terminal edges.
    g.add_grid_tedges(nodeids, unary_terms * 0, unary_terms)

    print("calling maxflow")
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    img2 = np.int_(np.logical_not(sgm))
    plt.imshow(img2[:, :, 0], cmap=plt.cm.gray, interpolation="nearest")

    lp = ImageLP()

    indices = lp.add_variables_array(
        shape=size_image, lower_bounds=0, upper_bounds=1, costs=unary_terms / coef_mul
    )

    ground_truth = img2
    ground_truth_indices = indices

    lp.add_pott_model(indices, coef_potts / coef_mul)
    return lp, ground_truth, ground_truth_indices, unary_terms


def run(display=True, image_size=50, coef_mul=500, coef_potts=0.5, max_duration=15):

    lp, ground_truth, ground_truth_indices, unary_terms = build_linear_program(
        image_size, coef_potts, coef_mul
    )

    print("solving")

    if display:
        fig = plt.figure()
        ax_image = fig.add_subplot(111)
        im = ax_image.imshow(
            unary_terms[:, :, 0] / coef_mul,
            cmap=plt.cm.Greys_r,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        fig_curves1 = plt.figure()
        ax_curves1 = fig_curves1.add_subplot(111)
        ax_curves1.set_xlabel("nb of iteration")
        ax_curves1.set_ylabel("distance_to_ground_truth")
        fig_curves2 = plt.figure()
        ax_curves2 = fig_curves2.add_subplot(111)
        ax_curves2.set_xlabel("duration")
        ax_curves2.set_ylabel("distance_to_ground_truth")

    def plot_solution(niter, solution, is_active_variable=None):
        image = solution[ground_truth_indices]
        # imwrite('ter%05d.png'%niter,solution[indices][:,:,0])
        # imwrite('diff_iter%05d.png'%niter,np.diff(solution[indices][:,:,0]))
        im.set_array(image[:, :, 0])
        # im.set_array(np.diff(image[:,:,0]))
        plt.draw()

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(2, 5, 1, title="graph cut")
        ax.imshow(ground_truth[:, :, 0], cmap=plt.cm.Greys_r, interpolation="none")
        ax.axis("off")

    # simplex much too slow for images larger than 20 by 20
    # LP2=copy.deepcopy(LP)
    # LP2.convert_to_one_sided_inequality_system()
    # sol1,elapsed=LP2.solve(method='ScipyLinProg',force_integer=False,get_timing=True,nb_iter=100,max_duration=10,ground_truth=ground_truth,ground_truth_indices=indices,plot_solution=None)

    solving_methods2 = list(solving_methods)
    for m in ["scipy_simplex", "scipy_interior_point"]:
        solving_methods2.remove(m)

    distance_to_ground_truth_curves = {}

    for i, method in enumerate(solving_methods2):
        print(
            "\n\n----------------------------------------------------------\nSolving LP using %s"
            % method
        )

        sol1, elapsed = lp.solve(
            method=method,
            get_timing=True,
            nb_iter=100000,
            max_duration=max_duration,
            ground_truth=ground_truth,
            ground_truth_indices=ground_truth_indices,
            plot_solution=None,
            nb_iter_plot=500,
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
            plt.gca().invert_yaxis()
            ax_curves2.legend()
            ax = fig.add_subplot(2, 5, i + 2, title=method)
            ax.imshow(
                sol1[ground_truth_indices][:, :, 0],
                cmap=plt.cm.Greys_r,
                interpolation="none",
                vmin=0,
                vmax=1,
            )
            ax.axis("off")
            plt.draw()

        distance_to_ground_truth_curves[method] = lp.distance_to_ground_truth

    if display:
        plt.tight_layout()
        # plt.figure()
        # plt.plot(LP.itrn_curve,LP.dopttime_curve,'g',label='admm')
        # plt.draw()
        # plt.show()
        print("done")
        plt.show()

    return distance_to_ground_truth_curves


if __name__ == "__main__":
    run()
