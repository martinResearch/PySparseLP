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
"""LP Solver based on a chambolle-pock algorithm."""
import time

import numpy as np

import scipy.ndimage
import scipy.sparse

from .tools import convert_to_standard_form_with_bounds


def chambolle_pock_ppd(
    c,
    a_eq,
    beq,
    a_ineq,
    b_lower,
    b_upper,
    lb,
    ub,
    x0=None,
    alpha=1,
    theta=1,
    nb_max_iter=100,
    callback_func=None,
    max_duration=None,
    save_problem=False,
    force_integer=False,
    nb_iter_plot=10,
):
    """Solve linear programming problem using chambolle-pock first order primal dual-method with preconditioning.

    Method adapted from Diagonal preconditioning for first order primal-dual algorithms in convex optimization
    by Thomas Pack and Antonin Chambolle
    the adaptation makes the code able to handle a more flexible specification of the LP problem
    (we could transform generic LPs into the equality form, but i am not sure the convergence would be the same)
    minimizes c.T*x
    such that
    a_eq*x=beq
    b_lower<= a_ineq*x<= b_upper
    lb<=x<=ub
    """
    assert scipy.sparse.issparse(a_ineq)
    start = time.clock()
    elapsed = start

    if a_eq.shape[0] == 0:
        a_eq = None
        beq = None

    if (a_ineq is not None) and (b_lower is not None):

        idskeep_upper = np.nonzero(b_upper != np.inf)[0]
        idskeep_lower = np.nonzero(b_lower != -np.inf)[0]
        if len(idskeep_lower) > 0 and len(idskeep_upper) > 0:
            a_ineq = scipy.sparse.vstack(
                (a_ineq[idskeep_upper, :], -a_ineq[idskeep_lower, :])
            ).tocsr()
        elif len(idskeep_lower) > 0:
            a_ineq = -a_ineq
        else:
            a_ineq = a_ineq
        b_ineq = np.hstack((b_upper[idskeep_upper], -b_lower[idskeep_lower]))
    else:
        b_ineq = b_upper

    use_vec_sparsity = False
    if x0 is not None:
        x = x0.copy()
    else:
        x = np.zeros(c.size)
    assert lb.size == c.size
    assert ub.size == c.size

    # save_problem=True
    if save_problem:
        with open("LP_problem2.pkl", "wb") as f:
            d = {
                "c": c,
                "a_eq": a_eq,
                "beq": beq,
                "a_ineq": a_ineq,
                "b_ineq": b_ineq,
                "lb": lb,
                "ub": ub,
            }
            import pickle

            pickle.dump(d, f)

    n = c.size
    use_standard_form = False
    if use_standard_form and (a_ineq is not None):
        c, a_eq, beq, lb, ub, x0 = convert_to_standard_form_with_bounds(
            c, a_eq, beq, a_ineq, b_ineq, lb, ub, x0
        )
        a_ineq = None

    use_column_preconditioning = True
    if use_column_preconditioning:
        # constructing the preconditioning diagonal matrices
        tmp = 0
        if a_eq is not None:
            print("a_eq shape=" + str(a_eq.shape))

            assert scipy.sparse.issparse(a_eq)
            assert a_eq.shape[1] == c.size
            assert a_eq.shape[0] == beq.size
            a_eq_copy = a_eq.copy()
            a_eq_copy.data = np.abs(a_eq_copy.data) ** (2 - alpha)
            sum_a_eq = np.ones((1, a_eq_copy.shape[0])) * a_eq_copy
            tmp = tmp + sum_a_eq
            # AeqT=a_eq.T
        if a_ineq is not None:
            print("a_ineq shape=" + str(a_ineq.shape))
            assert scipy.sparse.issparse(a_ineq)
            assert a_ineq.shape[1] == c.size
            assert a_ineq.shape[0] == b_ineq.size
            a_ineq_copy = a_ineq.copy()
            a_ineq_copy.data = np.abs(a_ineq_copy.data) ** (2 - alpha)
            sum_a_ineq = np.ones((1, a_ineq_copy.shape[0])) * a_ineq_copy
            tmp = tmp + sum_a_ineq
            # AineqT=a_ineq.T
        if a_eq is None and a_ineq is None:
            x = np.zeros_like(lb)
            x[c > 0] = lb[c > 0]
            x[c < 0] = ub[c < 0]
            return x
        tmp[tmp == 0] = 1
        diag_t = 1 / tmp[0, :]
        # T = scipy.sparse.diags(diag_t[None, :], [0]).tocsr()
    else:
        scipy.sparse.eye(len(x))
        diag_t = np.ones(x.shape)
    if a_eq is not None:
        a_eq_copy = a_eq.copy()
        a_eq_copy.data = np.abs(a_eq_copy.data) ** (alpha)
        sum_a_eq = a_eq_copy * np.ones((a_eq_copy.shape[1]))
        tmp = sum_a_eq
        tmp[tmp == 0] = 1
        diag_sigma_eq = 1 / tmp
        sigma_eq = scipy.sparse.diags([diag_sigma_eq], [0]).tocsr()
        y_eq = np.zeros(a_eq.shape[0])
        del a_eq_copy
        del sum_a_eq
    if a_ineq is not None:
        a_ineq_copy = a_ineq.copy()
        a_ineq_copy.data = np.abs(a_ineq_copy.data) ** (alpha)
        sum_a_ineq = a_ineq_copy * np.ones((a_ineq_copy.shape[1]))
        tmp = sum_a_ineq
        tmp[tmp == 0] = 1
        diag_sigma_ineq = 1 / tmp
        sigma_ineq = scipy.sparse.diags([diag_sigma_ineq], [0]).tocsr()
        y_ineq = np.zeros(a_ineq.shape[0])
        del a_ineq_copy
        del sum_a_ineq

    # some cleaning
    del tmp

    # del diagSigma
    # del diag_t

    # iterations
    # AeqT=AeqT.tocsc()
    # AineqT=AineqT.tocsc()
    x3 = x

    best_integer_solution_energy = np.inf
    best_integer_solution = None
    niter = 0
    while niter < nb_max_iter:

        # Update the primal variables
        d = c
        if a_eq is not None:
            if use_vec_sparsity:
                yeq_sparse = scipy.sparse.coo_matrix(y_eq).T
                d = (
                    d + (yeq_sparse * a_eq).toarray().ravel()
                )  # faster when few constraint are activated
            else:
                d = d + y_eq * a_eq
                # d+=y_eq*a_eq# strangely this does not work, give wrong results

        if a_ineq is not None:
            if use_vec_sparsity:
                y_ineq_sparse = scipy.sparse.coo_matrix(y_ineq).T
                d = (
                    d + (y_ineq_sparse * a_ineq).toarray().ravel()
                )  # faster when few constraint are activated
            else:
                d = d + y_ineq * a_ineq
                # d+=y_ineq*a_ineq

        # x2=x-T*d
        x2 = x - diag_t * d
        np.maximum(x2, lb, x2)
        np.minimum(x2, ub, x2)
        # x2=np.maximum(x2,lb)
        # x2=np.minimum(x2,ub)
        x3_prev = x3
        x3 = (1 + theta) * x2 - theta * x
        diff_x3 = x3_prev - x3
        x = x2
        if use_vec_sparsity:
            x3_sparse = scipy.sparse.coo_matrix(x3).T
        if a_eq is not None:
            if use_vec_sparsity:
                r_eq = (a_eq * x3_sparse).toarray().ravel() - beq
            else:
                r_eq = (a_eq * x3) - beq
        if a_ineq is not None:
            if use_vec_sparsity:
                r_ineq = (a_ineq * x3_sparse).toarray().ravel() - b_ineq
            else:
                r_ineq = (a_ineq * x3) - b_ineq

        if niter % nb_iter_plot == 0:

            elapsed = time.clock() - start
            if (max_duration is not None) and elapsed > max_duration:
                break
            energy1 = c.dot(x)

            # x4 is obtained my minimizing with respect to the primal variable while keeping the langrangian coef fix , which give a lower bound on the optimal solution
            # energy2 is the lower bound
            # energy1  is the value of the lagrangian at the current (hopefull saddle) point
            # on problem is that the minimization with respect to the primal variables may actually lead to infintely negative lower bounds...
            x4 = (
                -d * 100000
            )  # pb : the curve is very dependant on that value which make this lower bound a bit useless
            x4 = np.maximum(x4, lb)
            x4 = np.minimum(x4, ub)

            x4 = lb.copy()
            x4[d < 0] = ub[d < 0]

            energy2 = c.dot(x4)
            max_violated_equality = 0
            max_violated_inequality = 0
            if a_eq is not None:
                energy1 += y_eq.T.dot(a_eq * x - beq)
                energy2 += y_eq.T.dot(a_eq * x4 - beq)
                max_violated_equality = np.max(np.abs(r_eq))
            if a_ineq is not None:
                energy1 += y_ineq.T.dot(a_ineq * x - b_ineq)
                energy2 += y_ineq.T.dot(a_ineq * x4 - b_ineq)
                max_violated_inequality = np.max(r_ineq)
            if force_integer:
                x_rounded = np.round(x)
            else:
                x_rounded = x
            energy_rounded = c.dot(x_rounded)
            if a_eq is not None:
                max_violated_equality_rounded = np.max(np.abs(a_eq * x_rounded - beq))
            else:
                max_violated_equality_rounded = 0
            max_violated_inequality = np.max(a_ineq * x_rounded - b_ineq)
            if max_violated_equality_rounded == 0 and max_violated_inequality <= 0:
                print(
                    "##########   found feasible solution with energy"
                    + str(energy_rounded)
                )
                if energy_rounded < best_integer_solution_energy:
                    best_integer_solution_energy = energy_rounded
                    best_integer_solution = x_rounded

            print(
                f"iter {niter} elapsed {elapsed:2.1f} seconds: energy1={energy1:f} energy2={energy2:f}"
                f" max_viol_eq={max_violated_inequality:1.3e} max_viol_ineq ={max_violated_equality:1.3e}"
                f" x3_sparsity ={np.mean(x3 == 0):1.2f} diff_x3_sparsity={np.mean(diff_x3==0):1.2f}"
            )

            if callback_func is not None:

                callback_func(
                    niter,
                    x,
                    energy1,
                    energy2,
                    elapsed,
                    max_violated_equality,
                    max_violated_inequality,
                )

        # Update the dual variables

        if a_eq is not None:
            y_eq = y_eq + sigma_eq * r_eq
            # y_eq=y_eq+diag_sigma_eq*r_eq
            # y_eq+=diag_sigma_eq*r_eq

        if a_ineq is not None:
            y_ineq = y_ineq + sigma_ineq * r_ineq
            # y_ineq+=diag_sigma_ineq*r_ineq
            np.maximum(y_ineq, 0, y_ineq)
            # y_ineq=np.maximum(y_ineq, 0)
        niter += 1
    if best_integer_solution is not None:
        best_integer_solution = best_integer_solution[:n]
    return x[:n], best_integer_solution
