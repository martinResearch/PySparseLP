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
"""LP solver using gradient ascend in the dual. ot efficient but use as baseline"""

import copy
import time

import numpy as np

import scipy.ndimage
import scipy.sparse


def exact_dual_line_search(direction, a, b, c_bar, upper_bounds, lower_bounds):

    assert isinstance(direction, scipy.sparse.csr.csr_matrix)
    d_a = direction * a
    alphas = -c_bar[d_a.indices] / d_a.data
    order = np.argsort(alphas)
    d_a_u = d_a.data * upper_bounds[d_a.indices]
    d_a_l = d_a.data * lower_bounds[d_a.indices]
    tmp1 = np.minimum(d_a_u[order], d_a_l[order])
    tmp2 = np.maximum(d_a_u[order], d_a_l[order])
    tmp3 = np.cumsum(tmp2[::-1])[::-1]
    tmp4 = np.cumsum(tmp1)
    derivatives = -(direction.dot(b)) * np.ones(alphas.size + 1)
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
    else:

        alpha_optim = alphas[order[k - 1]]
    return alpha_optim


def dual_gradient_ascent(
    x,
    lp,
    nb_max_iter=1000,
    callback_func=None,
    y_eq=None,
    y_ineq=None,
    max_duration=None,
    nb_iter_plot=1,
):
    """Gradient ascent in the dual."""
    np.random.seed(0)
    start = time.clock()
    # convert to slack form (augmented form)
    lp2 = copy.deepcopy(lp)
    lp = None
    # LP2.convert_to_slack_form()
    assert (lp2.b_lower is None) or np.max(lp2.b_lower) == -np.inf
    # y_ineq=None
    # LP2.convertTo
    # LP2.convert_to_one_sided_inequality_system()
    # LP2.upper_bounds=np.minimum(10000,LP2.upper_bounds)
    # LP2.lower_bounds=np.maximum(-10000,LP2.lower_bounds)
    # y0=None
    if y_eq is None:
        y_eq = np.zeros(lp2.a_equalities.shape[0])
        y_eq = -np.random.rand(y_eq.size)
    else:
        y_eq = y_eq.copy()
    # y_ineq=None
    if y_ineq is None:
        if not (lp2.a_inequalities is None):
            y_ineq = np.zeros(lp2.a_inequalities.shape[0])
            y_ineq = np.abs(np.random.rand(y_ineq.size))
    else:
        y_ineq = y_ineq.copy()
    # assert (LP2.b_lower is None)

    def get_optim_x(y_eq, y_ineq):
        c_bar = lp2.costsvector.copy()
        if lp2.a_equalities is not None:
            c_bar += y_eq * lp2.a_equalities
        if lp2.a_inequalities is not None:
            c_bar += y_ineq * lp2.a_inequalities
        x = np.zeros(lp2.costsvector.size)
        x[c_bar > 0] = lp2.lower_bounds[c_bar > 0]
        x[c_bar < 0] = lp2.upper_bounds[c_bar < 0]
        x[c_bar == 0] = 0.5 * (lp2.lower_bounds + lp2.upper_bounds)[c_bar == 0]
        # t=np.random.rand(np.sum(c_bar==0))
        # x[c_bar==0]=t*LP2.lower_bounds[c_bar==0]+(1-t)*LP2.upper_bounds[c_bar==0]

        return c_bar, x

    def evaluate(y_eq, y_ineq):
        c_bar, x = get_optim_x(y_eq, y_ineq)

        # E=-y_eq.dot(LP2.b_equalities)-y_ineq.dot(LP2.b_upper)+np.sum(x*c_bar)
        # LP2.costsvector.dot(x)+y_ineq.dot(LP2.a_inequalities*x-LP2.b_upper)
        energy = np.sum(
            np.minimum(c_bar * lp2.upper_bounds, c_bar * lp2.lower_bounds)[c_bar != 0]
        )
        if lp2.a_equalities is not None:
            energy -= y_eq.dot(lp2.b_equalities)
        if lp2.a_inequalities is not None:
            energy -= y_ineq.dot(lp2.b_upper)
        return energy

    # x[c_bar==0]=0.5

    # alpha_i= vector containing the step lengths that lead to a sign change on any of the gradient component
    # when incrementing y[i]
    #
    print("iter %d energy %f" % (0, evaluate(y_eq, y_ineq)))

    prev_energy = evaluate(y_eq, y_ineq)
    if prev_energy == -np.inf:
        print("initial dual point not feasible, you could bound all variables")
        c_bar, x = get_optim_x(y_eq, y_ineq)
        return x, y_eq, y_ineq
    niter = 0
    while niter < nb_max_iter:
        c_bar, x = get_optim_x(y_eq, y_ineq)
        if lp2.a_inequalities is not None:
            y_ineq_prev = y_ineq.copy()
            max_violation = np.max(lp2.a_inequalities * x - lp2.b_upper)
            sum_violation = np.sum(np.maximum(lp2.a_inequalities * x - lp2.b_upper, 0))
            np.sum(np.maximum(lp2.a_inequalities * x - lp2.b_upper, 0))
            if (niter % nb_iter_plot) == 0:
                print(
                    "iter %d energy %f max violation %f sum_violation %f"
                    % (niter, prev_energy, max_violation, sum_violation)
                )

            grad_y_ineq = lp2.a_inequalities * x - lp2.b_upper

            grad_y_ineq[y_ineq_prev <= 0] = np.maximum(
                grad_y_ineq[y_ineq_prev <= 0], 0
            )  # not sure it is correct to do that here
            if np.sum(grad_y_ineq < 0) > 0:

                grad_y_ineq_sparse = scipy.sparse.csr.csr_matrix(grad_y_ineq)
                coef_length_ineq = exact_dual_line_search(
                    grad_y_ineq_sparse,
                    lp2.a_inequalities,
                    lp2.b_upper,
                    c_bar,
                    lp2.upper_bounds,
                    lp2.lower_bounds,
                )
                # y_ineq_prev+coef_length*grad_y>0
                assert coef_length_ineq >= 0
                maxstep_ineq = np.min(
                    y_ineq_prev[grad_y_ineq < 0] / -grad_y_ineq[grad_y_ineq < 0]
                )
                coef_length_ineq = min(coef_length_ineq, maxstep_ineq)
                # if False:
                # y2=y_ineq.copy()
                # alphasgrid=np.linspace(coef_length*0.99,coef_length*1.01,1000)
                # vals=[]
                # for alpha in alphasgrid:
                # y2=y_ineq+alpha*grad_y
                # vals.append(evaluate(y_eq,y2))
                # plt.plot(alphasgrid,vals,'.')

                # coef_length=0.001/(niter+2000000)
                # coef_length=min(0.01/(niter+200000),maxstep)
                y_ineq = y_ineq_prev + coef_length_ineq * grad_y_ineq
                # assert(np.min(y_ineq)>=-1e-8)
                y_ineq = np.maximum(y_ineq, 0)

        if lp2.a_equalities is not None and lp2.a_equalities.shape[0] > 0:

            y_eq_prev = y_eq.copy()
            max_violation = np.max(np.abs(lp2.a_equalities * x - lp2.b_equalities))
            sum_violation = np.sum(np.abs(lp2.a_equalities * x - lp2.b_equalities))
            if (niter % nb_iter_plot) == 0:
                print(
                    "iter %d energy %f max violation %f sum_violation %f"
                    % (niter, prev_energy, max_violation, sum_violation)
                )

            grad_y_eq = lp2.a_equalities * x - lp2.b_equalities
            if np.any(grad_y_eq):
                grad_y_eq_sparse = scipy.sparse.csr.csr_matrix(grad_y_eq)
                coef_length_eq = exact_dual_line_search(
                    grad_y_eq_sparse,
                    lp2.a_equalities,
                    lp2.b_equalities,
                    c_bar,
                    lp2.upper_bounds,
                    lp2.lower_bounds,
                )
                # y_ineq_prev+coef_length*grad_y>0
                assert coef_length_eq >= 0

                y_eq = y_eq_prev + coef_length_eq * grad_y_eq

        # while True:
        # y_ineq=y_ineq_prev+coef_length*grad_y
        # newEnergy=evaluate(y_eq,y_ineq)
        # if newEnergy< prevEnergy:
        # coef_length=coef_length*0.5
        # print 'reducing step length'
        # else:
        # coef_length=coef_length*1.5
        # break
        new_energy = evaluate(y_eq, y_ineq)
        prev_energy = new_energy
        elapsed = time.clock() - start
        if callback_func is not None and niter % 100 == 0:
            callback_func(niter, x, 0, 0, elapsed, 0, 0)

        if (max_duration is not None) and elapsed > max_duration:
            break
        niter += 1

    print("done")
    return x, y_eq, y_ineq
