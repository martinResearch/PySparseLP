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
"""LP Solver inspired byt the Chambolle-Pock algorithm with a speedup using only a subset of the constraints at each iteration. Doe snot converge well"""
import copy
import time


import numpy as np

import scipy.ndimage
import scipy.sparse

from .tools import convert_to_standard_form_with_bounds

# @profile


def chambolle_pock_ppdas(
    lp,
    x0=None,
    alpha=1,
    theta=1,
    nb_iter=100,
    callback_func=None,
    max_time=None,
    nb_iter_plot=300,
    save_problem=False,
    frequency_update_active_set=20,
):

    # method adapted from
    # Diagonal preconditioning for first order primal-dual algorithms in convex optimization
    # by Thomas Pack and Antonin Chambolle
    # the adaptatio makes the code able to handle a more flexible specification of the LP problem
    # (we could transform genric LPs into the equality form , but i am note sure the convergence would be the same)
    # minimizes c.T*x
    # such that
    # a_eq*x=beq
    # b_lower<= a_ineq*x<= b_upper               assert(scipy.sparse.issparse(a_ineq))

    # lb<=x<=ub
    # callback_func=None

    lp2 = copy.deepcopy(lp)
    lp2.convert_to_one_sided_inequality_system()

    lp2.upper_bounds = np.minimum(10000, lp2.upper_bounds)
    lp2.lower_bounds = np.maximum(-10000, lp2.lower_bounds)

    c = lp2.costsvector
    a_eq = lp2.a_equalities
    if a_eq.shape[0] == 0:
        a_eq = None
        y_eq = None
    beq = lp2.b_equalities
    a_ineq = lp2.a_inequalities
    b_lower = lp2.b_lower
    b_upper = lp2.b_upper
    b_ineq = b_upper
    lb = lp2.lower_bounds
    ub = lp2.upper_bounds
    # c,a_eq,beq,a_ineq,b_lower,b_upper,lb,ub
    assert b_lower is None
    assert lb.size == c.size
    assert ub.size == c.size

    start = time.clock()
    elapsed = start

    use_vec_sparsity = False
    if x0 is not None:
        x = x0.copy()
    else:
        x = np.zeros(c.size)

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
    if use_standard_form:
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
            assert a_ineq.shape[0] == b_upper.size
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
        # sigma_eq = scipy.sparse.diags([diag_sigma_eq], [0]).tocsr()
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
        # sigma_ineq = scipy.sparse.diags([diag_sigma_ineq], [0]).tocsr()
        y_ineq = np.zeros(a_ineq.shape[0])
        del a_ineq_copy
        del sum_a_ineq

    # some cleaning
    del tmp

    # del diagSigma
    # del diag_t

    # iterations
    # AeqT=AeqT.tocsc()callback_func
    # AineqT=AineqT.tocsc()
    x3 = x

    best_integer_solution_energy = np.inf
    best_integer_solution = None

    list_active_variables = np.arange(x.size)
    is_active_variable = np.ones(x.shape, dtype=np.bool)
    if a_ineq is not None:
        is_active_inequality_constraint = np.ones(a_ineq.shape[0], dtype=np.bool)
        list_active_inequality_constraints = np.arange(a_ineq.shape[0])
    if a_eq is not None:
        is_active_equality_constraint = np.ones(a_eq.shape[0], dtype=np.bool)
        list_active_equality_constraints = np.arange(a_eq.shape[0])
        a_eq_csc = a_eq.tocsc()
        a_eq_csr = a_eq.tocsr()
        r_eq = (a_eq * x) - beq
        r_eq_active = r_eq[list_active_equality_constraints]
        sub_a_eq_csr = a_eq_csr[list_active_equality_constraints, :]
        sub_a_eq_csc2 = sub_a_eq_csr.tocsc()[:, list_active_variables]

        sub_a_eq_csr2 = sub_a_eq_csc2.tocsr()
    d = c.copy()
    if a_ineq is not None:
        a_ineq_csc = a_ineq.tocsc()
        a_ineq_csr = a_ineq.tocsr()
        r_ineq = (a_ineq * x) - b_ineq
        # subAeqCSC=a_eq_csc[list_active_equality_constraints,:]
        # suba_ineq_csc=a_ineq_csc[list_active_inequality_constraints,:]
        r_ineq_active = r_ineq[list_active_inequality_constraints]
        sub_a_ineq_csr = a_ineq_csr[list_active_inequality_constraints, :]
        sub_a_ineq_csc2 = sub_a_ineq_csr.tocsc()[:, list_active_variables]
        sub_a_ineq_csr2 = sub_a_ineq_csc2.tocsr()
        diag_sigma_ineq_active = diag_sigma_ineq[list_active_inequality_constraints]
        active_y_ineq = y_ineq[list_active_inequality_constraints]

    x_active = x[list_active_variables]
    lb_active = lb[list_active_variables]
    ub_active = ub[list_active_variables]

    diag_t_active = diag_t[list_active_variables]
    d_active = d[list_active_variables]

    x3 = x
    x3_active = x3[list_active_variables]

    diff_active_y_eq = None
    diff_active_y_ineq = None

    for i in range(nb_iter):

        # Update he primal variables

        if a_eq is not None:
            if use_vec_sparsity:
                yeq_sparse = scipy.sparse.coo_matrix(y_eq).T
                d = (
                    d + (yeq_sparse * a_eq).toarray().ravel()
                )  # faster when few constraint are activated
            else:
                if i == 0:
                    # d=d+y_eq*a_eq
                    d_active = d_active + y_eq * sub_a_eq_csr2
                else:
                    # sparse_diff_y_eq=scipy.sparse.csr_matrix((diff_active_y_eq, list_active_equality_constraints, [0,list_active_equality_constraints.size]), shape=(1,a_eq.shape[0]))
                    # d=d+diff_active_y_eq*a_eq[list_active_equality_constraints,:]

                    # increment=sparse_diff_y_eq*a_eq_csr
                    # d=d+increment.toarray().ravel()
                    # d[increment.indices]=d[increment.indices]+increment.data#  does numpoy exploit the fact that second term is sparse ?
                    # d=d+diff_active_y_eq*sub_a_eq_csr
                    d_active = (
                        d_active + diff_active_y_eq * sub_a_eq_csr2
                    )  # will be usefull when few active variables
                # d+=y_eq*a_eq# strangley this does not work, give wrong results

        if a_ineq is not None:
            if use_vec_sparsity:
                y_ineq_sparse = scipy.sparse.coo_matrix(y_ineq).T
                d = (
                    d + (y_ineq_sparse * a_ineq).toarray().ravel()
                )  # faster when few constraint are activated
            else:
                if i == 0:
                    # d=d+y_ineq*a_ineq
                    d_active = d_active + y_ineq * sub_a_ineq_csr2
                else:

                    # d=d+diff_active_y_ineq*a_ineq[list_active_inequality_constraints,:]# does numpoy exploit the fact that second term is sparse ?
                    # sparse_diff_y_ineq=scipy.sparse.csr_matrix((diff_active_y_ineq, list_active_inequality_constraints, [0,list_active_inequality_constraints.size]), shape=(1,a_ineq.shape[0]))
                    # increment=sparse_diff_y_ineq*a_ineq_csr
                    # d=d+increment.toarray().ravel()
                    # d[increment.indices]=d[increment.indices]+increment.data
                    # d=d+diff_active_y_ineq*sub_a_ineq_csr
                    d_active = d_active + diff_active_y_ineq * sub_a_ineq_csr2
                # d+=y_ineq*a_ineq

        # update list active variables
        # ideally find largest values in primal steps diag_t*d, should ne do that at each step, or have a
        # structure that allows to maintaint a list of larges values
        if True and i > 0 and i % frequency_update_active_set == 0:
            x[list_active_variables] = x_active
            x3[list_active_variables] = x3_active
            y_ineq[list_active_inequality_constraints] = active_y_ineq
            # update d for the variables that where inactive

            d = c + y_ineq * a_ineq_csr
            if a_eq is not None:
                d += y_eq * a_eq_csr

            # tmp=np.minimum(x-lb,np.maximum(diag_t*d,0))+np.minimum(ub-x,np.maximum(-diag_t*d,0))

            if True:
                tmp = np.abs(diag_t * d)
                is_active_variable = tmp > 1e-6
                is_active_variable[(diag_t * d > 1e-3) & (x - lb == 0)] = False
                is_active_variable[(diag_t * d < -1e-3) & (ub - x == 0)] = False
                (list_active_variables,) = np.nonzero(
                    is_active_variable
                )  # garde les 10 % le plus larges
            d_active = d[list_active_variables]

            # update list active constraints
            # ideally find largest values in duals steps diag_sigma_ineq*r_ineq
            if a_eq is not None:
                r_eq = (a_eq_csc * x3) - beq
            r_ineq = (a_ineq_csc * x3) - b_ineq
            # tmp=np.abs(diag_sigma_ineq*r_ineq)*((y_ineq>0) | (diag_sigma_ineq*r_ineq>0 ))
            # list_active_inequality_constraints,=np.nonzero(tmp>np.percentile(tmp, 10))
            is_active_inequality_constraint = (r_ineq > -0.2) | (y_ineq > 0)
            (list_active_inequality_constraints,) = np.nonzero(
                is_active_inequality_constraint
            )

            if a_eq is not None:
                tmp = np.abs(diag_sigma_eq * r_eq)
                is_active_equality_constraint = tmp > 1e-6
                (list_active_equality_constraints,) = np.nonzero(
                    is_active_equality_constraint
                )
                nb_active_equality_constraints = np.sum(is_active_equality_constraint)
                percent_active_equality_constraint = 100 * np.mean(
                    is_active_equality_constraint
                )
            else:
                nb_active_equality_constraints = 0
                percent_active_equality_constraint = 0

            # subAeqCSC=a_eq_csc[list_active_equality_constraints,:]
            # subAeqCSC=a_eq_csr[list_active_equality_constraints,:].tocsc()
            if a_eq is not None:
                r_eq_active = r_eq[list_active_equality_constraints]
            # suba_ineq_csc=a_ineq_csc[list_active_inequality_constraints,:]
            # suba_ineq_csc=a_ineq_csr[list_active_inequality_constraints,:].tocsc()
            if a_eq is not None:
                sub_a_eq_csr = a_eq_csr[list_active_equality_constraints, :]
                sub_a_eq_csc2 = sub_a_eq_csr.tocsc()[:, list_active_variables]
                sub_a_eq_csr2 = sub_a_eq_csc2.tocsr()
            sub_a_ineq_csr = a_ineq_csr[list_active_inequality_constraints, :]

            sub_a_ineq_csc2 = sub_a_ineq_csr.tocsc()[:, list_active_variables]

            sub_a_ineq_csr2 = sub_a_ineq_csc2.tocsr()
            r_ineq_active = r_ineq[list_active_inequality_constraints]
            if i % nb_iter_plot == 0:
                print(
                    "%d active variables %d  active inequalities %d active equalities"
                    % (
                        np.sum(is_active_variable),
                        np.sum(is_active_inequality_constraint),
                        nb_active_equality_constraints,
                    )
                )

                print(
                    "%f percent of active variables %f percent active inequalities %f percent active equalities"
                    % (
                        100 * np.mean(is_active_variable),
                        100 * np.mean(is_active_inequality_constraint),
                        percent_active_equality_constraint,
                    )
                )
            x_active = x[list_active_variables]
            x3_active = x3[list_active_variables]
            lb_active = lb[list_active_variables]
            ub_active = ub[list_active_variables]
            diag_t_active = diag_t[list_active_variables]
            diag_sigma_ineq_active = diag_sigma_ineq[list_active_inequality_constraints]
            active_y_ineq = y_ineq[list_active_inequality_constraints]
        # x2=x-T*d

        new_active_x = x_active - diag_t_active * d_active
        # np.maximum(x2,lb,x2)
        # np.minimum(x2,ub,x2)

        np.maximum(new_active_x, lb_active, new_active_x)
        np.minimum(new_active_x, ub_active, new_active_x)

        x3_prev = x3_active
        x3_active = (1 + theta) * new_active_x - theta * x_active  # smoothing ?
        # diff_x3=x3_prev-x3
        diff_active_x3 = x3_active - x3_prev
        x_active = new_active_x
        # sparse_diff_x=scipy.sparse.csc_matrix((diff_active_x, list_active_variables, [0,list_active_variables.size]), shape=(x.size,1))

        if use_vec_sparsity:
            x3_sparse = scipy.sparse.coo_matrix(x3).T
        if a_eq is not None:
            if use_vec_sparsity:
                r_eq = (a_eq * x3_sparse).toarray().ravel() - beq
            else:

                # r_eq=r_eq+(a_eq[:,list_active_variables]*diff_active_x)# can use sparisity in diff_x3
                # r_eq=r_eq+a_eq*sparse_diff_x
                # increment=subAeqCSC*sparse_diff_x # to do : update only the active constraints residuals
                # r_eq_active[increment.indices]=r_eq_active[increment.indices]+increment.data

                r_eq_active += sub_a_eq_csc2 * diff_active_x3
                # r_eq=r_eq+increment.toarray().ravel()

        if a_ineq is not None:
            if use_vec_sparsity:
                r_ineq = (a_ineq * x3_sparse).toarray().ravel() - b_ineq
            else:

                # r_ineq=r_ineq+(a_ineq[:,list_active_variables]*diff_active_x)
                # r_ineq=r_ineq+a_ineq*sparse_diff_x
                # increment=suba_ineq_csc*sparse_diff_x# to do : update only the active constraints residuals
                # increment=sub_a_ineq_csc2*diff_active_x
                # r_ineq=r_ineq+increment.toarray().ravel()
                # r_ineq_active[increment.indices]=r_ineq_active[increment.indices]+increment.data
                r_ineq_active += sub_a_ineq_csc2 * diff_active_x3

        if i > 0 and i % nb_iter_plot == 0:
            x[list_active_variables] = x_active
            if a_ineq is not None:
                y_ineq[list_active_inequality_constraints] = active_y_ineq
                r_ineq = (a_ineq_csc * x) - b_ineq
            if a_eq is not None:
                r_eq = (a_eq_csc * x) - beq

            prev_elapsed = elapsed
            elapsed = time.clock() - start
            mean_iter_period = (elapsed - prev_elapsed) / 10
            if (max_time is not None) and elapsed > max_time:
                break
            energy1 = c.dot(x)

            # x4 is obtained my minimizing with respect to the primal variable while keeping the langrangian coef fix , which give a lower bound on the optimal solution
            # energy2 is the lower bound
            # energy1  is the value of the lagrangian at the curretn (hopefull sadle) point
            x4 = np.zeros(lb.size)

            # c_bar=LP2.costsvector.copy()
            # if not LP2.a_equalities	 is None and LP2.a_equalities.shape[0]>0:
            # c_bar+=y_eq*LP2.a_equalities
            # if not LP2.a_inequalities is None:
            # c_bar+=y_ineq*LP2.a_inequalities

            d[list_active_variables] = d_active
            c_bar = d.copy()
            x4[c_bar > 0] = lb[c_bar > 0]
            x4[c_bar < 0] = ub[c_bar < 0]
            x4[c_bar == 0] = 0

            energy2 = np.sum(x4 * c_bar)

            max_violated_equality = 0
            max_violated_inequality = 0
            if a_eq is not None:
                energy1 += y_eq.T.dot(a_eq * x - beq)
                energy2 -= y_eq.dot(beq)
                max_violated_equality = np.max(np.abs(r_eq))
            if a_ineq is not None:
                energy1 += y_ineq.T.dot(a_ineq * x - b_ineq)
                energy2 -= y_ineq.dot(b_ineq)
                max_violated_inequality = np.max(r_ineq)

            x_rounded = np.round(x)
            # x_rounded=greedy_round(x,c,a_eq,beq,a_ineq,np.full(b_ineq.shape,-np.inf),b_ineq,lb.copy(),ub.copy(),callback_func=callback_func)

            energy_rounded = c.dot(x_rounded)
            if a_eq is not None:
                nb_violated_equality_rounded = np.sum(np.abs(a_eq * x_rounded - beq))
            else:
                nb_violated_equality_rounded = 0
            if a_ineq is not None:
                nb_violated_inequality_rounded = np.sum(
                    np.maximum(a_ineq * x_rounded - b_ineq, 0)
                )
            else:
                nb_violated_inequality_rounded = 0

            if (
                nb_violated_equality_rounded == 0
                and nb_violated_inequality_rounded == 0
            ):
                print(
                    "##########   found feasible solution with energy"
                    + str(energy_rounded)
                )
                if energy_rounded < best_integer_solution_energy:
                    best_integer_solution_energy = energy_rounded
                    best_integer_solution = x_rounded

            print(
                "iter"
                + str(i)
                + ": energy1= "
                + str(energy1)
                + " energy2="
                + str(energy2)
                + " elapsed "
                + str(elapsed)
                + " second"
                + " max violated inequality:"
                + str(max_violated_inequality)
                + " max violated equality:"
                + str(max_violated_equality)
                + "mean_iter_period="
                + str(mean_iter_period)
                + "rounded : %f ineq %f eq"
                % (nb_violated_inequality_rounded, nb_violated_equality_rounded)
            )
            # 'y_eq has '+str(100 * np.mean(y_eq==0))+' % of zeros '+\
            #    'y_ineq has '+str(100 * np.mean(y_ineq==0))+' % of zeros '+\

            if callback_func is not None:

                callback_func(
                    i,
                    x,
                    energy1,
                    energy2,
                    elapsed,
                    max_violated_equality,
                    max_violated_inequality,
                    is_active_variable=is_active_variable,
                )

        # Update the dual variables

        if a_eq is not None:
            diff_active_y_eq = (
                diag_sigma_eq[list_active_equality_constraints] * r_eq_active
            )
            y_eq[list_active_equality_constraints] = (
                y_eq[list_active_equality_constraints] + diff_active_y_eq
            )

            # y_eq=y_eq+diag_sigma_eq*r_eq
            # y_eq+=diag_sigma_eq*r_eq

        if a_ineq is not None:
            # active_y_ineq=y_ineq[list_active_inequality_constraints]
            new_active_y_ineq = active_y_ineq + diag_sigma_ineq_active * r_ineq_active
            new_active_y_ineq = np.maximum(new_active_y_ineq, 0)
            diff_active_y_ineq = new_active_y_ineq - active_y_ineq
            # np.mean(diff_active_y_ineq!=0) often give me 0.05 on the facade , can i use that for more speedups ?
            active_y_ineq = new_active_y_ineq
            # y_ineq[list_active_inequality_constraints]=active_y_ineq

            # y_ineq+=diag_sigma_ineq*r_ineq
            # np.maximum(y_ineq, 0,y_ineq)

    return x[:n], best_integer_solution
