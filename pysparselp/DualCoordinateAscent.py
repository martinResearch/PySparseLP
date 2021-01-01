# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------
# Copyright © 2016 Martin de la Gorce <martin[dot]delagorce[hat]gmail[dot]com>

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------
"""LP solver using alternated  coordinate ascend on the dual."""

import copy
import time

import numpy as np

import scipy.ndimage
import scipy.sparse

from .DualGradientAscent import exact_dual_line_search
from .constraintPropagation import greedy_round

from numba import jit
from numba.experimental import jitclass
import numba


def get_optim_x(lp2, y_eq, y_ineq, tiemethod="round", x0=None, upate_x_cbar_zero=True):
    c_bar = lp2.costsvector.copy()
    if lp2.a_equalities is not None:
        c_bar += y_eq * lp2.a_equalities
    if lp2.a_inequalities is not None:
        c_bar += y_ineq * lp2.a_inequalities
    if x0 is None:
        x = np.zeros(lp2.costsvector.size)
    else:
        x = x0
    x[c_bar > 0] = lp2.lower_bounds[c_bar > 0]
    x[c_bar < 0] = lp2.upper_bounds[c_bar < 0]

    if upate_x_cbar_zero:
        if tiemethod == "round":
            x[c_bar == 0] = (
                lp2.lower_bounds
                + np.random.rand(len(lp2.upper_bounds))
                * (lp2.upper_bounds - lp2.lower_bounds)
            )[c_bar == 0]
        elif tiemethod == "center":
            x[c_bar == 0] = 0.5 * (lp2.lower_bounds + lp2.upper_bounds)[c_bar == 0]
        else:
            print("unkown tie method %s" % tiemethod)
            raise
        x[(c_bar == 0) & np.isinf(lp2.lower_bounds)] = lp2.upper_bounds[
            (c_bar == 0) & np.isinf(lp2.lower_bounds)
        ]
        x[(c_bar == 0) & np.isinf(lp2.upper_bounds)] = lp2.lower_bounds[
            (c_bar == 0) & np.isinf(lp2.upper_bounds)
        ]
        x[
            (c_bar == 0) & np.isinf(lp2.upper_bounds) & np.isinf(lp2.lower_bounds)
        ] = 0  # could take any arbitrary value
        # x[(c_bar==0) & (LP2.costsvector>0)]=LP2.lower_bounds[(c_bar==0) & (LP2.costsvector>0)]
        # x[(c_bar==0) & (LP2.costsvector<0)]=LP2.upper_bounds[(c_bar==0) & (LP2.costsvector<0)]
    return c_bar, x


def evaluate(lp2, y_eq, y_ineq):
    c_bar, x = get_optim_x(lp2, y_eq, y_ineq)
    energy = -y_eq.dot(lp2.b_equalities) - y_ineq.dot(lp2.b_upper) + np.sum(x * c_bar)
    # LP2.costsvector.dot(x)+y_ineq.dot(LP2.a_inequalities*x-LP2.b_upper)
    energy = (
        -y_eq.dot(lp2.b_equalities)
        - y_ineq.dot(lp2.b_upper)
        + np.sum(
            np.minimum(c_bar * lp2.upper_bounds, c_bar * lp2.lower_bounds)[c_bar != 0]
        )
    )
    return energy


@jit(nopython=True)
def exact_coordinate_line_search(
    lower_bounds, upper_bounds, a_ineq_col_i, b_ineq_i, c_bar
):
    alphas = -c_bar[a_ineq_col_i.indices] / a_ineq_col_i.data
    order = np.argsort(alphas)
    a_ineq_upper = a_ineq_col_i.data * upper_bounds[a_ineq_col_i.indices]
    a_ineq_lower = a_ineq_col_i.data * lower_bounds[a_ineq_col_i.indices]
    tmp1 = np.minimum(a_ineq_upper[order], a_ineq_lower[order])
    tmp2 = np.maximum(a_ineq_upper[order], a_ineq_lower[order])
    tmp3 = np.cumsum(tmp2[::-1])[::-1]
    tmp4 = np.cumsum(tmp1)
    derivatives = -b_ineq_i * np.ones(alphas.size + 1)
    derivatives[:-1] += tmp3
    derivatives[1:] += tmp4

    # tmp=np.abs(Ai.data[order])*(LP2.lower_bounds[Ai.indices[order]]-LP2.upper_bounds[Ai.indices[order]])
    # derivatives= -LP2.b_equalities[i]-np.sum(AiL[Ai.data>0])-np.sum(AiU[Ai.data<0])\
    # +np.hstack(([0],np.cumsum(tmp)))

    k = np.searchsorted(-derivatives, 0)
    if derivatives[k] == 0 and k < len(order):
        t = np.random.rand()
        alpha_optim = (
            t * alphas[order[k]] + (1 - t) * alphas[order[k - 1]]
        )  # maybe could draw and random value in the interval ?
        # alpha_optim=alphas[order[k-1]]
    else:
        alpha_optim = alphas[order[k - 1]]
    return alpha_optim


@jit(nopython=True)
def equalities_iteration(
    x,
    c_bar,
    costsvector,
    a_equalities,
    b_equalities,
    lower_bounds,
    upper_bounds,
    y_eq,
    y_ineq,
):

    grad_y_eq = a_equalities.mul(x) - b_equalities
    # list_i = np.nonzero(grad_y_eq)[0]
    for i in range(len(grad_y_eq)):

        # i=32
        # print evaluate(y)
        # if False:
        #     import matplotlib.pyplot as plt

        #     y2 = y_eq.copy()
        #     vals = []
        #     alphas_grid = np.linspace(-1, 1, 1000)
        #     for alpha in alphas_grid:
        #         y2[i] = y_eq[i] + alpha
        #         vals.append(evaluate(lp2, y2, y_ineq))
        #     plt.plot(alphas_grid, vals, ".")
        #     deriv = np.diff(vals) / np.diff(alphas_grid)
        #     plt.plot(alphas_grid[:-1], deriv, ".")

        a_eq_col_i = a_equalities.row(i)
        # c_bar=LP2.costsvector+y_eq*LP2.a_equalities+y_ineq*LP2.a_inequalities
        alpha_optim = exact_coordinate_line_search(
            lower_bounds, upper_bounds, a_eq_col_i, b_equalities[i], c_bar
        )
        prev_y_eq = y_eq[i]
        y_eq[i] += alpha_optim
        diff_y_eq = y_eq[i] - prev_y_eq
        c_bar[a_eq_col_i.indices] += diff_y_eq * a_eq_col_i.data


spec = [
    ("shape", numba.types.UniTuple(numba.int32, 2)),  # a simple scalar field
    ("data", numba.float64[:]),  # an array field
    ("indices", numba.int32[:]),  # an array field
    ("indptr", numba.int32[:]),  # an array field
]


@jitclass(spec)
class csr_matrix:
    """minimal sparse matrix implementation with
    only few methods implemented that can be used with numba
    as numba-scipy does not support sparse matrices yet"""
    def __init__(self, shape, data, indices, indptr):
        self.shape = shape
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def row(self, i):
        data = self.data[self.indptr[i] : self.indptr[i + 1]]
        indices = self.indices[self.indptr[i] : self.indptr[i + 1]]
        indptr = self.indptr[i : i + 2]
        return csr_matrix((1, self.shape[1]), data, indices, indptr)

    def mul(self, other):
        nrow = self.shape[0]
        result = np.zeros((nrow,), dtype=np.float64)
        for i in range(nrow):
            data = self.data[self.indptr[i] : self.indptr[i + 1]]
            indices = self.indices[self.indptr[i] : self.indptr[i + 1]]
            result[i] = np.sum(data * other[indices])
        return result


@jit(nopython=True)
def inequalities_iteration(
    x,
    c_bar,
    costsvector,
    a_inequalities,
    b_upper,
    lower_bounds,
    upper_bounds,
    y_eq,
    y_ineq,
):

    grad_y_ineq = a_inequalities.mul(x) - b_upper
    grad_y_ineq[y_ineq <= 0] = np.maximum(grad_y_ineq[y_ineq <= 0], 0)  #

    for i in np.nonzero(grad_y_ineq)[0]:

        a_ineq_col_i = a_inequalities.row(i)
        # if False:
        #     c_bar, x = get_optim_x(lp2, y_eq, y_ineq)
        #     grad_y_ineq = lp2.a_inequalities * x - lp2.b_upper
        #     grad_y_ineq[y_ineq <= 0] = np.maximum(grad_y_ineq[y_ineq <= 0], 0)  #
        #     alpha_optim = exact_coordinate_line_search(
        #         lp2, a_ineq_col_i, lp2.b_upper[i], c_bar
        #     )
        #     y2 = y_ineq.copy()
        #     vals = []
        #     alphas_grid = np.linspace(-4, 0, 1000)
        #     for alpha in alphas_grid:
        #         y2[i] = y_ineq[i] + alpha
        #         vals.append(evaluate(lp2, y_eq, y2))
        #     plt.plot(alphas_grid, vals, ".")
        #     deriv = np.diff(vals) / np.diff(alphas_grid)
        #     plt.plot(alphas_grid[:-1], deriv, ".")

        # c_bar=LP2.costsvector+y_eq*LP2.a_equalities+y_ineq*LP2.a_inequalities
        alpha_optim = exact_coordinate_line_search(
            lower_bounds, upper_bounds, a_ineq_col_i, b_upper[i], c_bar
        )

        # prev_energy=evaluate(lp2, y_eq,y_ineq)
        prev_y_ineq = y_ineq[i]
        y_ineq[i] += alpha_optim
        y_ineq[i] = max(y_ineq[i], 0)
        diff_y_ineq = y_ineq[i] - prev_y_ineq
        c_bar[a_ineq_col_i.indices] += diff_y_ineq * a_ineq_col_i.data
        # new_energy=evaluate(lp2, y_eq,y_ineq)
        # assert(new_energy>=prev_energy-1e-5)
        # assert(np.max(y_ineq)<=0)


def dual_coordinate_ascent(
    x,
    lp,
    nb_max_iter=20,
    callback_func=None,
    y_eq=None,
    y_ineq=None,
    max_time=None,
    nb_iter_plot=1,
    use_rounding=False,
    use_over_relaxation=True
):
    """Solve LP using coordinate ascent in the dual.

    Method from 'An algorithm for large scale 0-1 integer
    programming with application to airline crew scheduling'
    we generalized it to the case where A is not 0-1 and
    the upper bounds can be greater than 1
    did not generalize and code the approximation method
    """
    alpha_relax = 0.05
    print(numba.typeof((5, 6)))
    np.random.seed(1)
    start = time.clock()
    # convert to slack form (augmented form)
    lp2 = copy.deepcopy(lp)
    lp = None
    # LP2.convert_to_slack_form()
    # LP2.convertTo
    lp2.convert_to_one_sided_inequality_system()

    # LP2.upper_bounds=np.minimum(10,LP2.upper_bounds)
    # LP2.lower_bounds=np.maximum(-10,LP2.lower_bounds)
    # y0=None
    if y_eq is None:
        y_eq = np.zeros(lp2.a_equalities.shape[0])
        # y_eq=-np.random.rand(y_eq.size)
    else:
        y_eq = y_eq.copy()
    # y_ineq=None
    if y_ineq is None:
        y_ineq = np.zeros(lp2.a_inequalities.shape[0])
        # y_ineq=np.abs(np.random.rand(y_ineq.size))
    else:
        y_ineq = y_ineq.copy()
        assert np.min(y_ineq) >= 0
    # assert (LP2.b_lower is None)

    # x[c_bar==0]=0.5

    # alpha_i= vector containing the step lengths that lead to a sign change on any of the gradient component
    # when incrementing y[i]
    #
    energy = evaluate(lp2, y_eq, y_ineq)

    print("iter %d energy %f" % (0, energy))
    c_bar, x = get_optim_x(lp2, y_eq, y_ineq)
    direction = np.zeros(y_ineq.shape)

    timeout = False
    niter = 0

    a_equalities = csr_matrix(
        lp2.a_equalities.shape,
        lp2.a_equalities.data,
        lp2.a_equalities.indices,
        lp2.a_equalities.indptr,
    )
    a_inequalities = csr_matrix(
        lp2.a_inequalities.shape,
        lp2.a_inequalities.data,
        lp2.a_inequalities.indices,
        lp2.a_inequalities.indptr,
    )

    while niter < nb_max_iter:
        if timeout:
            break

        y_ineq_prev = y_ineq.copy()

        c_bar = lp2.costsvector + y_eq * lp2.a_equalities + y_ineq * lp2.a_inequalities
        equalities_iteration(
            x,
            c_bar,
            lp2.costsvector,
            a_equalities,
            lp2.b_equalities,
            lp2.lower_bounds,
            lp2.upper_bounds,
            y_eq,
            y_ineq,
        )

        if timeout:
            break

        c_bar = lp2.costsvector + y_eq * lp2.a_equalities + y_ineq * lp2.a_inequalities
        new_energy = evaluate(lp2, y_eq, y_ineq)
        eps = 1e-10
        if new_energy + eps < energy:
            print(f"not expected decrease {new_energy - energy}")

        energy = new_energy

        c_bar, x = get_optim_x(lp2, y_eq, y_ineq, x0=None, upate_x_cbar_zero=True)

        inequalities_iteration(
            x,
            c_bar,
            lp2.costsvector,
            a_inequalities,
            lp2.b_upper,
            lp2.lower_bounds,
            lp2.upper_bounds,
            y_eq,
            y_ineq,
        )

        if timeout:
            break
        new_energy = evaluate(lp2, y_eq, y_ineq)
        if new_energy + eps < energy:
            print(f"not expected decrease {new_energy - energy}")

        c_bar, x = get_optim_x(
            lp2, y_eq, y_ineq, tiemethod="center", x0=None, upate_x_cbar_zero=True
        )

        if use_rounding and new_energy < energy + 1e-10:
            x[c_bar == 0] = 0.5 * (lp2.lower_bounds + lp2.upper_bounds)[
                c_bar == 0
            ] + 0.1 * np.sign(lp2.costsvector[c_bar == 0])
            order = np.argsort(np.abs(x - 0.5))
            fixed = c_bar != 0
            xr, valid = greedy_round(
                x, lp2, callback_func=None, maxiter=30, order=order, fixed=fixed
            )
            lp2.costsvector.dot(xr)
            x = xr

        energy_upper_bound = lp2.costsvector.dot(x)

        elapsed = time.clock() - start
        if (niter % nb_iter_plot) == 0:
            max_violation = max(
                np.max(lp2.a_inequalities * x - lp2.b_upper),
                np.max(np.sum(np.abs(lp2.a_equalities * x - lp2.b_equalities))),
            )
            sum_violation = np.sum(
                np.maximum(lp2.a_inequalities * x - lp2.b_upper, 0)
            ) + np.sum(np.abs(lp2.a_equalities * x - lp2.b_equalities))
            print(
                "iter %d time %3.1f dual energy %f, primal %f max violation %f sum_violation %f"
                % (
                    niter,
                    elapsed,
                    new_energy,
                    energy_upper_bound,
                    max_violation,
                    sum_violation,
                )
            )
            if max_violation == 0:

                print(
                    "found feasible primal solution with energy %f" % energy_upper_bound
                )
                if energy_upper_bound == new_energy:
                    print("found optimal solution , stop")
                    break
                if energy_upper_bound < new_energy:
                    print("not expected")
                if new_energy < energy + 1e-10:
                    print("will not find better solution , stop")
                    break

        energy = new_energy
        if callback_func is not None:
            callback_func(niter, x, 0, 0, elapsed, 0, 0)
        if use_over_relaxation:
            diff = y_ineq - y_ineq_prev
            direction = direction * (1-alpha_relax) + alpha_relax * diff
            direction[np.abs(direction) < 1e-10] = 0
            direction = scipy.sparse.csr.csr_matrix(direction)
            c_bar, x = get_optim_x(
                lp2, y_eq, y_ineq, tiemethod="center", x0=None, upate_x_cbar_zero=True
            )
            coef_length = exact_dual_line_search(
                direction,
                lp2.a_inequalities,
                lp2.b_upper,
                c_bar,
                lp2.upper_bounds,
                lp2.lower_bounds,
            )
            y_ineq = np.array(y_ineq + coef_length * direction).flatten()
            y_ineq = np.maximum(y_ineq, 0)
            new_energy = evaluate(lp2, y_eq, y_ineq)
            eps = 1e-10
            if new_energy + eps < energy:
                print(f"not expected decrease n {new_energy - energy}")

            energy = new_energy
            # y_ineq=y_ineq+*0.1
            # y_ineq=np.maximum(y_ineq, 0)
            #print("iter %d energy %f" % (niter, evaluate(lp2, y_eq, y_ineq)))

        if (max_time is not None) and elapsed > max_time:
            timeout = True
            break
        niter += 1
    max_violation = max(
        np.max(lp2.a_inequalities * x - lp2.b_upper),
        np.max(np.sum(np.abs(lp2.a_equalities * x - lp2.b_equalities))),
    )
    sum_violation = np.sum(
        np.maximum(lp2.a_inequalities * x - lp2.b_upper, 0)
    ) + np.sum(np.abs(lp2.a_equalities * x - lp2.b_equalities))
    print(
        "iter %d time %3.1f dual energy %f, primal %f max violation %f sum_violation %f"
        % (niter, elapsed, new_energy, energy_upper_bound, max_violation, sum_violation)
    )
    return x, y_eq, y_ineq
