"""Example using a pott image model that can be exactly solved with graphcut."""

import matplotlib.pyplot as plt

import maxflow  # pip install PyMaxflow

import numpy as np

from pysparselp.SparseLP import SparseLP, solving_methods
from pysparselp.examples.benchmark_methods import benchmark_methods


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

    def set_pixel_indices(self, indices):
        self.pixel_indices = indices

    def display_solution(self, ax, sol):
        ax.imshow(sol[self.pixel_indices].squeeze(axis=2))


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

    lp = ImageLP()

    indices = lp.add_variables_array(
        shape=size_image, lower_bounds=0, upper_bounds=1, costs=unary_terms / coef_mul
    )
    lp.set_pixel_indices(indices)

    lp.add_pott_model(indices, coef_potts / coef_mul)
    return lp


def run(
    display=True,
    image_size=50,
    coef_mul=500,
    coef_potts=0.5,
    max_duration=5,
    nb_iter_plot=100,
):

    lp = build_linear_program(image_size, coef_potts, coef_mul)

    solving_methods2 = list(solving_methods)
    solving_methods2.remove("scipy_interior_point")  # too slow

    distance_to_ground_truth_curves = benchmark_methods(
        lp,
        solving_methods2,
        display_solution_func=lp.display_solution,
        max_duration=max_duration,
        nb_iter_plot=nb_iter_plot,
        display=display,
    )
    return distance_to_ground_truth_curves


if __name__ == "__main__":
    run()
