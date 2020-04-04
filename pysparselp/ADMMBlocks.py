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


import copy
import numpy as np
import time
from scipy import sparse
import scipy.ndimage

from pysparselp.tools import (
    preconditionConstraints,
    convertToStandardFormWithBounds,
    chrono,
    check_decrease,
)

# import  scikits.sparse.cholmod


def LP_admmBlockDecomposition(
    c,
    Aeq,
    beq,
    Aineq,
    b_lower,
    b_upper,
    lb,
    ub,
    x0=None,
    gamma_ineq=0.7,
    nb_iter=100,
    callbackFunc=None,
    max_time=None,
    use_preconditionning=True,
    useLU=True,
    nb_iter_plot=10,
):
    # simple ADMM method with an approximate resolution of a quadratic subproblem using conjugate gradient
    # inspiredy by Boyd's paper on ADMM
    # Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
    # the difference with LP_admm is that the linear quality constrainrs Aeq*beq are enforced during the resolution
    # of the subproblem instead of beeing enforced through multipliers
    n = c.size
    start = time.clock()
    elapsed = start
    if x0 == None:
        x0 = np.zeros(c.size)
    # if Aeq!=None:
    # Aeq,beq=preconditionConstraints(Aeq,beq,alpha=2)
    # if Aineq!=None:# it seem important to do this preconditionning before converting to standard form
    # Aineq,b_lower,b_upper=preconditionConstraints(Aineq,b_lower,b_upper,alpha=2)

    c, Aeq, beq, lb, ub, x0 = convertToStandardFormWithBounds(
        c, Aeq, beq, Aineq, b_lower, b_upper, lb, ub, x0
    )
    x = x0

    xp = x.copy()
    xp = np.maximum(xp, lb)
    xp = np.minimum(xp, ub)

    # trying some preconditioning
    use_preconditionning_rows = False  # left preconditioning seems not to change anything if not used in combination with use_preconditionning_cols  as each subproblem is solved exactly.
    if use_preconditionning_rows:
        Aeq, beq = preconditionConstraints(Aeq, beq, alpha=2)

    # global right preconditioning
    use_preconditionning_cols = False
    if use_preconditionning_cols:
        R, c, Aeq, beq, lb, ub, x0 = preconditionLPRight(
            c, Aeq, beq, lb, ub, x0, alpha=3
        )
    else:
        R = sparse.eye(Aeq.shape[1])

    luMs = []
    nb_used = np.zeros(x.shape)
    list_block_ids = []
    # Aeq.blocks=[(0,Aeq.blocks[5][1]),(Aeq.blocks[6][0],Aeq.blocks[11][1])]
    # Aeq.blocks=[(0,Aeq.blocks[-1][1])]

    beqs = []
    usesparseLU = True
    xv = []
    # for idblock in range(nb_blocks):
    ch = chrono()

    mergegroupes = []
    mergegroupes = [[k] for k in range(len(Aeq.blocks))]
    # mergegroupes.append(np.arange(len(Aeq.blocks)))#single block

    # mergegroupes.append([0,1,2,3, 4, 5])
    # mergegroupes.append([6,7,8,9,10,11])
    # mergegroupes.append([12,13,14])
    # mergegroupes.append([0,1,2,3, 4, 5,12])
    # mergegroupes.append([6,7,8,9,10,11,13])
    # mergegroupes.append([14])

    # mergegroupes.append([0,1,2,3,4,5,12,13,])
    # mergegroupes.append([6,7,8,9,10,11,14,15])
    # mergegroupes.append([0,1,2,3,4,5])
    # mergegroupes.append([6,7,8,9,10,11])
    # mergegroupes.append([12,13])
    # mergegroupes.append([14,15])
    # mergegroupes.append([12,13,14,15])
    nb_blocks = len(mergegroupes)

    if False:
        idRows = np.hstack(
            [np.arange(Aeq.blocks[g][0], Aeq.blocks[g][1] + 1) for g in mergegroupe]
        )

        # we want to cluster the constraints such that the submatrix for each cluster has a sparse cholesky decomposition
        # we need to reach a tradeoff between the number of cluster (less variables copies) and the sparsity of the cholesky decompositions
        # each cluster should have a small tree width ?
        # can we do an incremental sparse cholseky an then  add one constraint to each cholesky at a time ?
        # is non sparse cholseky decomposition  (without permutation)incremental ?
        # adding a factor to a block is good if does not add too many new variables to the block and does not augment the treewidth of the block ?
        # we can do incremental merging of blocks
        # marge is good if the interection of the variables used by the two block is large (remove copies) (fast to test)
        # and if the operation does not increase the cholseky density or graph tree width too much

        subA = Aeq[idRows, :]
        t = np.array(np.abs(subA).sum(axis=0)).ravel()
        ids = np.nonzero(t)[0]
        list_block_ids.append(ids)

        subA2 = subA[:, ids]
        # precompute the LU factorizartion of the matrix that needs to be inverted for the block
        M = scipy.sparse.vstack(
            (
                scipy.sparse.hstack(
                    (
                        gamma_ineq * scipy.sparse.eye(subA2.shape[1], subA2.shape[1]),
                        subA2.T,
                    )
                ),
                scipy.sparse.hstack(
                    (subA2, scipy.sparse.csc_matrix((subA2.shape[0], subA2.shape[0])))
                ),
            )
        ).tocsr()
        LU = scikits.sparse.cholmod.cholesky(M.tocsc(), mode="simplicial")
        print(
            "the sparsity ratio between Chol(M) and the  matrix M  is +"
            + str(LU.L().nnz / float(M.nnz))
        )

        print(
            "connected components M" + str(scipy.sparse.csgraph.connected_components(M))
        )

        # scipy.sparse.csgraph.connected_components(subA.T*subA)
        FactorConnections = subA * subA.T
        print(
            "connected components F"
            + str(scipy.sparse.csgraph.connected_components(FactorConnections))
        )
        ST = scipy.sparse.csgraph.minimum_spanning_tree(FactorConnections)

    for idblock, mergegroupe in enumerate(mergegroupes):
        # find the indices of the variables used by the block
        idRows = np.hstack(
            [np.arange(Aeq.blocks[g][0], Aeq.blocks[g][1] + 1) for g in mergegroupe]
        )
        subA = Aeq[idRows, :]
        t = np.array(np.abs(subA).sum(axis=0)).ravel()
        ids = np.nonzero(t)[0]
        list_block_ids.append(ids)
        # increment th number of time each variable is copied
        nb_used[ids] += 1
        subA2 = subA[:, ids]
        # precompute the LU factorizartion of the matrix that needs to be inverted for the block
        M = scipy.sparse.vstack(
            (
                scipy.sparse.hstack(
                    (
                        gamma_ineq * scipy.sparse.eye(subA2.shape[1], subA2.shape[1]),
                        subA2.T,
                    )
                ),
                scipy.sparse.hstack(
                    (subA2, scipy.sparse.csc_matrix((subA2.shape[0], subA2.shape[0])))
                ),
            )
        ).tocsr()
        if usesparseLU:

            ch.tic()
            LU = scipy.sparse.linalg.splu(M.tocsc())
            print(ch.toc())
        else:
            ch.tic()
            LU = scikits.sparse.cholmod.cholesky(M.tocsc(), mode="simplicial")
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
                + str(idblock)
                + " is +"
                + str(LU.L().nnz / float(M.nnz))
                + " took "
                + str(factorization_duration)
                + "seconds to factorize"
            )

            # LU.__=LU.solve_A
            # M2=convertToPySparseFormat(M)
            # LU = umfpack.factorize(M2, strategy="UMFPACK_STRATEGY_SYMMETRIC")
            # print "nnz per line :"+str(LU.nnz/float(M2.shape[0]) )

        xv.append(np.empty(M.shape[1], dtype=float))
        luMs.append(LU)
        beqs.append(beq[idRows])
        pass

    def L(x, xp, lambda_ineq):
        E = c.dot(xp)
        for idblock in range(nb_blocks):
            diff = x[idblock] - xp[list_block_ids[idblock]]
            E += 0.5 * gamma_ineq * np.sum((diff) ** 2) + lambda_ineq[idblock].dot(diff)
        return E

    i = 0

    x = [x0[list_block_ids[i]] for i in range(nb_blocks)]
    lambda_ineq = [np.zeros(list_block_ids[i].shape) for i in range(nb_blocks)]

    check = check_decrease(tol=1e-10)
    dpred = [np.zeros(list_block_ids[i].shape) for i in range(nb_blocks)]
    alpha = 1.95  # relaxation paramter should be in [0,2] , 1.95 seems to be often a good choice

    while i <= nb_iter:
        # solve the penalized problems with respect to each copy x
        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))
        # check.set(L(x, xp,lambda_ineq))
        for idblock in range(nb_blocks):
            y = np.hstack(
                (
                    gamma_ineq * xp[list_block_ids[idblock]] - lambda_ineq[idblock],
                    beqs[idblock],
                )
            )
            if usesparseLU:
                xv[idblock] = luMs[idblock].solve(y)
            else:
                xv[idblock] = luMs[idblock].solve_A(y)

                # luMs[idblock].solve(y,xv[idblock])
            x[idblock] = (
                alpha * xv[idblock][: x[idblock].shape[0]]
                + (1 - alpha) * xp[list_block_ids[idblock]]
            )

            # check.add(L(x, xp,lambda_ineq))
        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))
        # solve the penalized problem with respect to xp
        # c-sum_idblock  gamma_ineq*(x_[idblock]-xp[list_block_ids[idblock]])-lambda_ineq[idblock]=0
        xp[nb_used > 0] = 0
        for idblock in range(nb_blocks):
            xp[list_block_ids[idblock]] += (
                x[idblock] + lambda_ineq[idblock] / gamma_ineq
            )  # change formula here

        xp = xp - c / gamma_ineq
        xp = xp / np.maximum(nb_used, 1)
        xp = np.maximum(xp, lb)
        xp = np.minimum(xp, ub)
        # check.add(L(x, xp,lambda_ineq))

        for idblock in range(nb_blocks):
            d = gamma_ineq * (x[idblock] - xp[list_block_ids[idblock]])
            # angle=np.sum(dpred[idblock]*d)/(np.sqrt(np.sum(dpred[idblock]**2))*+np.sqrt(np.sum(d**2)))
            # print angle
            # dpred[idblock]=d.copy()
            lambda_ineq[idblock] = lambda_ineq[idblock] + d
            # lambda_ineq[idblock]=lambda_ineq[idblock]+(1+max(0,angle))*d # trying some naive heuristic speedup but not working :(

        if i % nb_iter_plot == 0:
            prev_elapsed = elapsed
            elapsed = time.clock() - start
            if elapsed > max_time:
                break

            energy1 = L(x, xp, lambda_ineq)
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
                + " elaspsed "
                + str(elapsed)
                + " second"
                + " max violated inequality:"
                + str(max_violated_inequality)
                + " max violated equality:"
                + str(max_violated_equality)
            )
            if not callbackFunc is None:
                callbackFunc(
                    i,
                    (R * xp)[0:n],
                    energy1,
                    energy2,
                    elapsed,
                    max_violated_equality,
                    max_violated_inequality,
                )
        i += 1

    return (R * xp)[0:n]
