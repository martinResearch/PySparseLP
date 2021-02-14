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
"""LP solver using the alternating direction method of multipliers (ADMM) method with a decomposition of the constraint matrix into blocks."""

import time

import numpy as np

import scipy.ndimage
from scipy import sparse

from .tools import (
    CheckDecrease,
    Chrono,
    convert_to_standard_form_with_bounds,
    precondition_constraints,
    precondition_lp_right,
)

# import  scikits.sparse.cholmod


def lp_admm_block_decomposition(
    c,
    a_eq,
    beq,
    a_ineq,
    b_lower,
    b_upper,
    lb,
    ub,
    x0=None,
    gamma_ineq=0.07,
    nb_iter=100,
    callback_func=None,
    max_duration=None,
    use_preconditioning=True,
    use_lu=True,
    nb_iter_plot=10,
):

    n = c.size
    start = time.clock()
    elapsed = start
    if x0 is None:
        x0 = np.zeros(c.size)
    # if a_eq!=None:
    # a_eq,beq=precondition_constraints(a_eq,beq,alpha=2)
    # if a_ineq!=None:# it seem important to do this preconditioning before converting to standard form
    # a_ineq,b_lower,b_upper=precondition_constraints(a_ineq,b_lower,b_upper,alpha=2)

    c, a_eq, beq, lb, ub, x0 = convert_to_standard_form_with_bounds(
        c, a_eq, beq, a_ineq, b_lower, b_upper, lb, ub, x0
    )
    x = x0

    xp = x.copy()
    xp = np.maximum(xp, lb)
    xp = np.minimum(xp, ub)

    # trying some preconditioning
    # left preconditioning seems not to change anything if not used in combination with use_preconditioning_cols as each subproblem is solved exactly.
    use_preconditioning_rows = False
    if use_preconditioning_rows:
        a_eq, beq = precondition_constraints(a_eq, beq, alpha=2)

    # global right preconditioning
    use_preconditioning_cols = False
    if use_preconditioning_cols:
        r, c, a_eq, beq, lb, ub, x0 = precondition_lp_right(
            c, a_eq, beq, lb, ub, x0, alpha=3
        )
    else:
        r = sparse.eye(a_eq.shape[1])

    lu_m_s = []
    nb_used = np.zeros(x.shape)
    list_block_ids = []
    # a_eq.blocks=[(0,a_eq.blocks[5][1]),(a_eq.blocks[6][0],a_eq.blocks[11][1])]
    # a_eq.blocks=[(0,a_eq.blocks[-1][1])]

    beqs = []
    usesparse_lu = True
    xv = []
    # for id_block in range(nb_blocks):
    ch = Chrono()

    merge_groups = []
    merge_groups = [[k] for k in range(len(a_eq.blocks))]

    nb_blocks = len(merge_groups)

    if False:
        import scikits.sparse

        id_rows = np.hstack(
            [np.arange(a_eq.blocks[g][0], a_eq.blocks[g][1] + 1) for g in merge_groups]
        )

        # we want to cluster the constraints such that the submatrix for each cluster has a sparse cholesky decomposition
        # we need to reach a tradeoff between the number of cluster (less variables copies) and the sparsity of the cholesky decompositions
        # each cluster should have a small tree width ?
        # can we do an incremental sparse cholesky and then  add one constraint to each cholesky at a time ?
        # is non sparse cholesky decomposition  (without permutation)incremental ?
        # adding a factor to a block is good if does not add too many new variables to the block and does not augment the treewidth of the block ?
        # we can do incremental merging of blocks
        # marge is good if the interection of the variables used by the two block is large (remove copies) (fast to test)
        # and if the operation does not increase the cholesky density or graph tree width too much

        sub_a = a_eq[id_rows, :]
        t = np.array(np.abs(sub_a).sum(axis=0)).ravel()
        ids = np.nonzero(t)[0]
        list_block_ids.append(ids)

        sub_a2 = sub_a[:, ids]
        # precompute the LU factorization of the matrix that needs to be inverted for the block
        m = scipy.sparse.vstack(
            (
                scipy.sparse.hstack(
                    (
                        gamma_ineq * scipy.sparse.eye(sub_a2.shape[1], sub_a2.shape[1]),
                        sub_a2.T,
                    )
                ),
                scipy.sparse.hstack(
                    (
                        sub_a2,
                        scipy.sparse.csc_matrix((sub_a2.shape[0], sub_a2.shape[0])),
                    )
                ),
            )
        ).tocsr()
        lu = scikits.sparse.cholmod.cholesky(m.tocsc(), mode="simplicial")
        print(
            "the sparsity ratio between Chol(M) and the  matrix M  is +"
            + str(lu.L().nnz / float(m.nnz))
        )

        print(
            "connected components M" + str(scipy.sparse.csgraph.connected_components(m))
        )

        # scipy.sparse.csgraph.connected_components(subA.T*subA)
        factor_connections = sub_a * sub_a.T
        print(
            "connected components F"
            + str(scipy.sparse.csgraph.connected_components(factor_connections))
        )
        spanning_tree = scipy.sparse.csgraph.minimum_spanning_tree(factor_connections)
        print(spanning_tree)

    for id_block, merge_group in enumerate(merge_groups):
        # find the indices of the variables used by the block
        id_rows = np.hstack(
            [np.arange(a_eq.blocks[g][0], a_eq.blocks[g][1] + 1) for g in merge_group]
        )
        sub_a = a_eq[id_rows, :]
        t = np.array(np.abs(sub_a).sum(axis=0)).ravel()
        ids = np.nonzero(t)[0]
        list_block_ids.append(ids)
        # increment th number of time each variable is copied
        nb_used[ids] += 1
        sub_a2 = sub_a[:, ids]
        # precompute the LU factorization of the matrix that needs to be inverted for the block
        m = scipy.sparse.vstack(
            (
                scipy.sparse.hstack(
                    (
                        gamma_ineq * scipy.sparse.eye(sub_a2.shape[1], sub_a2.shape[1]),
                        sub_a2.T,
                    )
                ),
                scipy.sparse.hstack(
                    (
                        sub_a2,
                        scipy.sparse.csc_matrix((sub_a2.shape[0], sub_a2.shape[0])),
                    )
                ),
            )
        ).tocsr()
        if usesparse_lu:

            ch.tic()
            lu = scipy.sparse.linalg.splu(m.tocsc())
            print(f"splu for block {id_block} took {ch.toc()} seconds")
        else:
            ch.tic()
            lu = scikits.sparse.cholmod.cholesky(m.tocsc(), mode="simplicial")
            # may fail when M is not positive definite , which sometimes occurs
            factorization_duration = ch.toc()
            # A=     scikits.sparse.cholmod.analyze(M.tocsc(),mode='simplicial')
            # P=scipy.sparse.coo_matrix((np.ones(A.P().size),(A.P(),np.arange(A.P().size))))
            # permuted=P*M*P.T
            # plt.imshow(permuted.todense())
            # A.P()

            # LU=     scikits.sparse.cholmod.cholesky(M.tocsc(),mode='supernodal')# gives me matrix is not positive definite errors..

            print(
                "the sparsity ratio between Chol(M) and the  matrix M for block"
                + str(id_block)
                + " is +"
                + str(lu.L().nnz / float(m.nnz))
                + " took "
                + str(factorization_duration)
                + "seconds to factorize"
            )

            # LU.__=LU.solve_A
            # M2=convert_to_py_sparse_format(M)
            # LU = umfpack.factorize(M2, strategy="UMFPACK_STRATEGY_SYMMETRIC")
            # print "nnz per line :"+str(LU.nnz/float(M2.shape[0]) )

        xv.append(np.empty(m.shape[1], dtype=float))
        lu_m_s.append(lu)
        beqs.append(beq[id_rows])
        pass

    def energy(x, xp, lambda_ineq):
        en = c.dot(xp)
        for id_block in range(nb_blocks):
            diff = x[id_block] - xp[list_block_ids[id_block]]
            en += 0.5 * gamma_ineq * np.sum((diff) ** 2) + lambda_ineq[id_block].dot(
                diff
            )
        return en

    i = 0

    x = [x0[list_block_ids[i]] for i in range(nb_blocks)]
    lambda_ineq = [np.zeros(list_block_ids[i].shape) for i in range(nb_blocks)]

    CheckDecrease(tol=1e-10)

    # relaxation parameter should be in [0,2] , 1.95 seems to be often a good choice
    alpha = 1.95

    while i <= nb_iter:
        # solve the penalized problems with respect to each copy x
        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))
        # check.set_value(L(x, xp,lambda_ineq))
        for id_block in range(nb_blocks):
            y = np.hstack(
                (
                    gamma_ineq * xp[list_block_ids[id_block]] - lambda_ineq[id_block],
                    beqs[id_block],
                )
            )
            if usesparse_lu:
                xv[id_block] = lu_m_s[id_block].solve(y)
            else:
                xv[id_block] = lu_m_s[id_block].solve_A(y)

                # luMs[id_block].solve(y,xv[id_block])
            x[id_block] = (
                alpha * xv[id_block][: x[id_block].shape[0]]
                + (1 - alpha) * xp[list_block_ids[id_block]]
            )

            # check.add_value(L(x, xp,lambda_ineq))
        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))
        # solve the penalized problem with respect to xp
        # c-sum_idblock  gamma_ineq*(x_[id_block]-xp[list_block_ids[id_block]])-lambda_ineq[id_block]=0
        xp[nb_used > 0] = 0
        for id_block in range(nb_blocks):
            xp[list_block_ids[id_block]] += (
                x[id_block] + lambda_ineq[id_block] / gamma_ineq
            )  # change formula here

        xp = xp - c / gamma_ineq
        xp = xp / np.maximum(nb_used, 1)
        xp = np.maximum(xp, lb)
        xp = np.minimum(xp, ub)
        # check.add(L(x, xp,lambda_ineq))

        for id_block in range(nb_blocks):
            d = gamma_ineq * (x[id_block] - xp[list_block_ids[id_block]])
            # angle=np.sum(dpred[id_block]*d)/(np.sqrt(np.sum(dpred[id_block]**2))*+np.sqrt(np.sum(d**2)))
            # print angle
            # dpred[id_block]=d.copy()
            lambda_ineq[id_block] = lambda_ineq[id_block] + d
            # lambda_ineq[id_block]=lambda_ineq[id_block]+(1+max(0,angle))*d # trying some naive heuristic speedup but not working :(

        if i % nb_iter_plot == 0:

            elapsed = time.clock() - start
            if elapsed > max_duration:
                break

            energy1 = energy(x, xp, lambda_ineq)
            energy2 = energy1

            max_violated_equality = 0
            max_violated_inequality = 0
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
            )
            if callback_func is not None:
                callback_func(
                    i,
                    (r * xp)[0:n],
                    energy1,
                    energy2,
                    elapsed,
                    max_violated_equality,
                    max_violated_inequality,
                )
        i += 1

    return (r * xp)[0:n]
