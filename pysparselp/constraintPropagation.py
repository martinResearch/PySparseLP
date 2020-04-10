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

"""Naive methods to round the solution of a continuous linear program solution to an integer solution using
- constraints propagation and backtracking
- greedy reduction of the number of violated constraints using local search
"""

import copy

import numpy as np

import scipy.sparse

try:

    from . import propagate_constraints as cython_propagate_constraints

    propagate_constraints_installed = True
except BaseException:
    print(
        "could not import propagate_constraints maybe the compilation did not work , will be slower"
    )
    propagate_constraints_installed = False


def check_constraints(i, x_r, mask, a_csr, a_csc, b_lower, b_upper):
    """Check that the variable i is not involed in any violated constraint."""
    violated = False
    constraints_to_check = np.nonzero(a_csc[:, i])[0]
    for j in constraints_to_check:
        line = a_csr[j, :]
        interval_d = 0
        interval_u = 0
        for k in range(line.indices.size):
            i = line.indices[k]
            v = line.data[k]

            if mask[i] > 0:
                interval_d += v * x_r[i]
                interval_u += v * x_r[i]
            elif v > 0:
                interval_u += v
            else:
                interval_d += v
        if interval_u < b_lower[j] or interval_d > b_upper[j]:
            violated = True
            break
    return violated


# @profile
def propagate_constraints(
    list_changed_var,
    x_l,
    x_u,
    a_csr,
    a_csc,
    b_lower,
    b_upper,
    back_ops,
    nb_iter=1000,
    use_cython=True,
):
    # may have similarities with the tightening method in http://www.davi.ws/doc/gondzio94presolve.pdf

    if propagate_constraints_installed and use_cython:

        # return cython_tools.propagate_constraints(list_changed_var,x_l,x_u,a_csr,a_csc,b_lower,b_upper,back_ops,nb_iter=nb_iter)
        return cython_propagate_constraints.propagate_constraints(
            list_changed_var,
            x_l,
            x_u,
            a_csc.indices,
            a_csr.indices,
            a_csr.indptr,
            a_csc.indptr,
            a_csr.data,
            b_lower,
            b_upper,
            back_ops,
            nb_iter=nb_iter,
        )
    tol = 1e-5  # to cope with small errors

    for _niter in range(nb_iter):
        # print '%d variable fixed '% np.sum(x_l==x_u)
        # list_changed_var=np.unique(list_changed_var)
        if len(list_changed_var) == 0:
            break

        list_constraints_to_check2 = []
        for i in list_changed_var:
            # to_add=np.nonzero(a_csc[:,i])[0]
            to_add2 = a_csc.indices[a_csc.indptr[i] : a_csc.indptr[i + 1]]
            # assert(np.all(to_add==to_add2))
            list_constraints_to_check2.append(to_add2)
        list_constraints_to_check2 = np.unique(np.hstack(list_constraints_to_check2))
        list_changed_var = []
        # list_constraints_to_check=np.arange(a_csr.shape[0])
        for j in list_constraints_to_check2:
            # line=a_csr[j,:]# very slow...
            # indices=line.indices
            # data=line.data
            indices = a_csr.indices[a_csr.indptr[j] : a_csr.indptr[j + 1]]
            data = a_csr.data[a_csr.indptr[j] : a_csr.indptr[j + 1]]

            interval_l = 0
            interval_u = 0
            for k in range(indices.size):
                i = indices[k]
                v = data[k]
                if v > 0:
                    interval_u += v * x_u[i]
                    interval_l += v * x_l[i]
                else:
                    interval_l += v * x_u[i]
                    interval_u += v * x_l[i]

            if interval_u < b_lower[j] or interval_l > b_upper[j]:
                return 0, j

            for k in range(indices.size):
                i = indices[k]
                v = data[k]
                if v > 0:

                    n_u = np.floor(tol + (b_upper[j] - interval_l + v * x_l[i]) / v)
                    n_l = np.ceil(-tol + (b_lower[j] - interval_u + v * x_u[i]) / v)
                else:
                    n_u = np.floor(tol + (b_lower[j] - interval_u + v * x_l[i]) / v)
                    n_l = np.ceil(-tol + (b_upper[j] - interval_l + v * x_u[i]) / v)

                changed = False
                if n_u < x_u[i]:
                    back_ops.append(
                        (1, i, x_u[i])
                    )  # save previous information for future backtracking
                    x_u[i] = n_u
                    changed = True
                if n_l > x_l[i]:
                    back_ops.append((0, i, x_l[i]))
                    x_l[i] = n_l
                    changed = True
                if changed:
                    list_changed_var.append(i)
                    # assert(j in list_constraints_to_check2)

    # print '%d variable fixed '% np.sum(x_l==x_u)
    return 1, None


def revert(back_ops, x_l, x_u):
    for t, i, v in reversed(back_ops):
        if t == 0:
            x_l[i] = v
        else:
            x_u[i] = v


# @profile


def greedy_round(
    x, lp, callback_func=None, maxiter=np.inf, order=None, fixed=None, display_func=None
):

    # save_arguments('greedy_round_test')
    if False:
        import pickle

        d = {"x": x, "lp": lp}
        with open("greedy_test.pkl", "wb") as f:
            pickle.dump(d, f)
    if callback_func is not None:
        callback_func(0, np.round(x), 0, 0, 0, 0, 0)
    lp2 = copy.copy(lp)
    lp2.convert_to_all_inequalities()
    assert lp2.a_equalities is None

    x_u = lp2.upper_bounds.copy()
    x_l = lp2.lower_bounds.copy()

    if fixed is not None:
        x_l[fixed] = x[fixed]
        x_u[fixed] = x[fixed]

    a_ineq = lp2.a_inequalities
    b_l = lp2.b_lower.copy()
    b_u = lp2.b_upper.copy()

    # callback_func(0,np.maximum(x_r.astype(np.float),0),0,0,0,0,0)
    a_ineq_csr = a_ineq.tocsr()
    a_ineq_csc = a_ineq.tocsc()
    if order is None:
        # sort from the less fractional to the most fractional
        # order=np.argsort(np.abs(x-np.round(x))+c*np.round(x))
        order = np.argsort(lp2.costsvector * (2 * np.round(x) - 1))
        # order=np.argsort(LP2.costsvector*np.round(x))
        # order=np.arange(x.size)
        # order=np.arange(x.size)[::-1]
    # x_r=np.full(x.size,-1,dtype=np.int32)
    x_r = x.copy()
    mask = np.zeros(x.size, dtype=np.int32)
    depth = 0
    nb_backtrack = 0
    # callback_func(0,x,0,0,0,0,0)

    valid, id_constraints = propagate_constraints(
        np.arange(a_ineq.shape[1]), x_l, x_u, a_ineq_csr, a_ineq_csc, b_l, b_u, []
    )
    if valid == 0:
        return x_r, valid

    # check that no constraint is violated
    back_ops = [[] for i in range(x.size)]
    niter = 0
    while depth < x.size:
        niter += 1

        if niter > maxiter:
            break

        # callback_func(0,x_l,0,0,0,0,0)
        # print depth

        id_var = order[depth]
        # print mask[order]
        if mask[id_var] == 2:
            mask[id_var] = 0
            revert(back_ops[depth], x_l, x_u)
            depth = depth - 1
            revert(back_ops[depth], x_l, x_u)
            print("step back to depth %d" % depth)
            if display_func is not None:
                display_func(x_r)
            continue

        if (
            x_u[id_var] == x_l[id_var]
        ):  # the variable is already fixed thanks to constraint propagation
            back_ops[depth] = []
            depth = depth + 1
            x_r[id_var] = x_u[id_var]
            mask[id_var] = 2
        elif mask[id_var] == 0:
            x_r[id_var] = np.round(x[id_var])
            if display_func is not None:
                display_func(x_r)
            mask[id_var] = 1
            back_ops[depth] = []
            back_ops[depth].append((1, id_var, x_u[id_var]))
            back_ops[depth].append((0, id_var, x_l[id_var]))
            x_u[id_var] = x_r[id_var]
            x_l[id_var] = x_r[id_var]

            # violated_eq=check_constraints(idvar,x_r,mask,Aeq_csr,Aeq_csc,beq,beq)
            # violated_ineq=check_constraints(idvar,x_r,mask,Aineq_csr,Aineq_csc,b_lower,b_upper)
            # violated=violated_eq | violated_ineq
            valid, id_constraints = propagate_constraints(
                [id_var], x_l, x_u, a_ineq_csr, a_ineq_csc, b_l, b_u, back_ops[depth]
            )
            x_r[x_l == x_u] = x_l[x_l == x_u]
            if display_func is not None:
                display_func(x_r)
            x_l[id_var]
            if valid:
                # valid,back_ops_init2=propagate_constraints(np.arange(A.shape[1]),x_l, x_u, a_csr, a_csc, b_l, b_u,[])
                # assert(len(back_ops_init2)==0)
                depth = depth + 1

            else:
                revert(back_ops[depth], x_l, x_u)

        elif mask[id_var] == 1:

            x_r[id_var] = 1 - round(x[id_var])
            back_ops[depth] = []
            back_ops[depth].append((1, id_var, x_u[id_var]))
            back_ops[depth].append((0, id_var, x_l[id_var]))
            x_u[id_var] = x_r[id_var]
            x_l[id_var] = x_r[id_var]

            mask[id_var] = 2
            valid, id_constraints = propagate_constraints(
                [id_var], x_l, x_u, a_ineq_csr, a_ineq_csc, b_l, b_u, back_ops[depth]
            )
            if valid:
                # valid,back_ops_init2=propagate_constraints(np.arange(A.shape[1]),x_l, x_u, a_csr, a_csc, b_l, b_u,[])
                # assert(len(back_ops_init2)==0)
                depth = depth + 1

            else:
                mask[id_var] = 0
                # callback_func(0,x_l,0,0,0,0,0)
                # name=LP.get_inequality_constraint_name_from_id(idcons)['name']
                # print 'constaint %d of type %s violated,steping back to depth %d'%(idcons,name,depth)

                x_l2 = x_l.copy() * 0.5
                x_l2[id_var] = 1
                # callback_func(0,x_l2,0,0,0,0,0)

                revert(back_ops[depth], x_l, x_u)
                depth = depth - 1
                nb_backtrack += 1
                revert(back_ops[depth], x_l, x_u)

                # raise # need a way to save the bound constraint to restore it

                # raise # need a way to save the bound constraint to restore it
    # callback_func(0,np.maximum(x_r.astype(np.float),0),0,0,0,0,0)
    valid = propagate_constraints(
        np.arange(a_ineq.shape[1]), x_l, x_u, a_ineq_csr, a_ineq_csc, b_l, b_u, []
    )
    # assert(valid)

    print("backtracked %d times" % nb_backtrack)
    print("energy after rounding =%f" % np.sum(x_r * lp.costsvector))

    return x_r, valid


def greedy_fix(x, lp, nb_max_iter=1000, callback_func=None, use_xor_moves=False):
    # decrease the constraints violation score using coordinate descent

    xr = np.round(x)

    lp2 = copy.copy(lp)

    # xors=np.nonzero(LP.b_lower==1)[0]

    # xors=np.nonzero(LP.b_equalities==1)[0]
    # assert(np.all(LP.a_equalities[xors,:].data==1))

    lp2.convert_to_all_inequalities()
    lp2.convert_to_one_sided_inequality_system()

    assert np.all(xr <= lp2.upper_bounds)
    assert np.all(xr >= lp2.lower_bounds)

    assert lp2.b_lower is None
    # compute the sum of the of violated constraints with the magnitude of the violation
    a_inequalities_csc = lp2.a_inequalities.tocsc()
    constraints_costs = np.ones(a_inequalities_csc.shape[0])
    # constraints_costs[:]=0.2
    xors = lp2.find_inequality_constraints_from_name("xors")
    for item in xors:
        constraints_costs[item["start"] : item["end"] + 1] = 1000
    # for item in xors:
    # print np.max(r_ineq_thresholded[item['start']:item['end']+1])
    r_ineq = lp2.a_inequalities * xr - lp2.b_upper
    r_ineq_thresholded = np.maximum(r_ineq, 0)
    score_ineq = np.sum(r_ineq_thresholded * constraints_costs)
    # test switching single variable
    # constraints_gradient=r_ineq_theholded*LP2.a_inequalities

    score_decrease = np.zeros(x.size)

    a_ineq = lp2.a_inequalities.copy()
    a_ineq.data = np.random.rand(a_ineq.data.size)
    to_check = np.nonzero(r_ineq_thresholded * a_ineq != 0)[0]
    check = False

    d_x = scipy.sparse.csc.csc_matrix(
        (1 - 2 * xr, (np.arange(xr.size), np.arange(xr.size))), (xr.size, xr.size)
    )
    dr_ineq_matrix = a_inequalities_csc * d_x

    if use_xor_moves:

        # adding xor moves
        # XorDx=
        xormoves = []
        xor_id_to_moves_interval = np.zeros(lp2.a_inequalities.shape[0])
        for xor_intervals in xors:
            for r in range(xor_intervals["start"], xor_intervals["end"] + 1):
                ids = lp2.a_inequalities[r, :].indices
                # data = LP2.a_inequalities[r, :].data
                assert len(ids) == 4

                vec = -xr[ids]
                xor_id_to_moves_interval[r] = len(xormoves)
                for i, _id in enumerate(ids):  # _id not used , is that a bug ?
                    vec2 = vec.copy()
                    vec2[i] += 1
                    xormoves.append((ids, vec2, r))

        xor_score_decrease = np.zeros(len(xormoves))
        for i, move in enumerate(xormoves):
            for j, idv in enumerate(move[0]):
                new_r_ineq = r_ineq[idv] + move[1][j]
                new_r_ineq_thresholded = np.maximum(new_r_ineq, 0)
                xor_score_decrease[i] += (
                    new_r_ineq_thresholded - r_ineq_thresholded[idv]
                ) * constraints_costs[idv]

    for _niter in range(nb_max_iter):
        if check:
            r_ineq = lp2.a_inequalities * xr - lp2.b_upper
            r_ineq_thresholded = np.maximum(r_ineq, 0)
            score_ineq = np.sum(r_ineq_thresholded * constraints_costs)

        dr_ineq_matrix = a_inequalities_csc * d_x[:, to_check]

        for j, i in enumerate(to_check):

            # dxi=1-2*xr[i]
            # dr_ineq=a_inequalities_csc[:,i]*dxi
            # dx=scipy.sparse.csc.csc_matrix(([1-2*xr[i]],([i],[0])),(xr.size,1))
            # dr_ineq=a_inequalities_csc*dx
            score_decrease[i] = 0
            dr_ineq = dr_ineq_matrix[:, j]
            assert dr_ineq.format == "csc"

            for j, idv in enumerate(dr_ineq.indices):
                new_r_ineq = r_ineq[idv] + dr_ineq.data[j]
                new_r_ineq_thresholded = np.maximum(new_r_ineq, 0)
                score_decrease[i] += (
                    new_r_ineq_thresholded - r_ineq_thresholded[idv]
                ) * constraints_costs[idv]

            if check:

                xr2 = xr.copy()
                xr2[i] = 1 - xr2[i]
                r_ineq2 = lp2.a_inequalities * xr2 - lp2.b_upper

                # r_ineq2b=dr_ineq.toarray().flatten()+r_ineq
                # np.max(np.abs(new_r_ineq_thresholded-r_ineq_thresholded2))
                # np.max(np.abs(new_r_ineq-r_ineq2))
                # np.max(np.abs(r_ineq2b-r_ineq2))
                r_ineq_thresholded2 = np.maximum(r_ineq2, 0)
                score_ineq2 = np.sum(r_ineq_thresholded2 * constraints_costs)
                score_decrease2 = score_ineq2 - score_ineq
                assert score_decrease2 == score_decrease[i]

            # if score_decrease[i]<0:
            # print "found move"

        # swith the variable that decreases the most the constraints violation score
        #
        if min(score_decrease) >= 0:
            print("could not find more moves")
            if callback_func is not None:
                callback_func(0, xr, 0, 0, 0, 0, 0)

            r_ineq = lp2.a_inequalities * xr - lp2.b_upper
            r_ineq_thresholded = np.maximum(r_ineq, 0)
            to_check = np.nonzero(r_ineq_thresholded * a_ineq != 0)[0]
            to_check = np.arange(xr.size)
            score_ineq2 = np.sum(r_ineq_thresholded * constraints_costs)
            assert score_ineq2 == score_ineq
            return xr

        i_best = np.argmin(score_decrease)
        # idbestxormove=np.argmin(xor_score_decrease)
        #

        # r_ineq=
        i = i_best
        # dxi=1-2*xr[i]

        # dr_ineq=a_inequalities_csc[:,i]*dxi
        dr_ineq = a_inequalities_csc * d_x[:, i]

        score_decrease_best = 0
        for j, idv in enumerate(dr_ineq.indices):
            r_ineq[idv] = r_ineq[idv] + dr_ineq.data[j]
            new_r_ineq_thresholded = np.maximum(r_ineq[idv], 0)
            score_decrease_best += (
                new_r_ineq_thresholded - r_ineq_thresholded[idv]
            ) * constraints_costs[idv]
            r_ineq_thresholded[idv] = new_r_ineq_thresholded

        assert np.abs(score_decrease_best - score_decrease[i_best]) < 1e-8

        score_ineq += score_decrease_best
        print(score_ineq)
        # xr[ibest]=1-xr[ibest]
        dx = d_x[:, i_best]
        xr[dx.indices] += dx.data
        if callback_func is not None:
            callback_func(0, xr, 0, 0, 0, 0, 0)

        # update switching score of variables that may have changed
        # tocheck=np.nonzero(dr_ineq.T*R!=0)[1]
        move_to_change = (dx.T * d_x).indices
        d_x[:, move_to_change] = scipy.sparse.csc.csc_matrix(
            (1 - 2 * xr[move_to_change], (move_to_change, np.arange(move_to_change.size))),
            (xr.size, move_to_change.size),
        )
        to_check = np.nonzero(dr_ineq.T * a_ineq * d_x != 0)[1]
