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


import numpy as np
import time
import scipy.sparse
import scipy.ndimage


def ChambollePockPPD(
    c,
    Aeq,
    beq,
    Aineq,
    b_lower,
    b_upper,
    lb,
    ub,
    x0=None,
    alpha=1,
    theta=1,
    nb_iter=100,
    callbackFunc=None,
    max_time=None,
    save_problem=False,
    force_integer=False,
    nb_iter_plot=10,
):
    # method adapted from
    # Diagonal preconditioning for first order primal-dual algorithms in convex optimization
    # by Thomas Pack and Antonin Chambolle
    # the adaptatio makes the code able to handle a more flexible specification of the LP problem
    # (we could transform genric LPs into the equality form , but i am note sure the convergence would be the same)
    # minimizes c.T*x
    # such that
    # Aeq*x=beq
    # b_lower<= Aineq*x<= b_upper               assert(scipy.sparse.issparse(Aineq))

    # lb<=x<=ub

    start = time.clock()
    elapsed = start

    if Aeq.shape[0] == 0:
        Aeq = None
        beq = None

    if (not Aineq is None) and (not b_lower is None):

        idskeep_upper = np.nonzero(b_upper != np.inf)[0]
        idskeep_lower = np.nonzero(b_lower != -np.inf)[0]
        if len(idskeep_lower) > 0 and len(idskeep_upper) > 0:
            Aineq = scipy.sparse.vstack(
                (Aineq[idskeep_upper, :], -Aineq[idskeep_lower, :])
            ).tocsr()
        elif len(idskeep_lower) > 0:
            Aineq = -Aineq
        else:
            Aineq = Aineq
        bineq = np.hstack((b_upper[idskeep_upper], -b_lower[idskeep_lower]))
    else:
        bineq = b_upper

    use_vec_sparsity = False
    if not x0 is None:
        x = xo.copy()
    else:
        x = np.zeros(c.size)
    assert lb.size == c.size
    assert ub.size == c.size

    # save_problem=True
    if save_problem:
        with open("LP_problem2.pkl", "wb") as f:
            d = {
                "c": c,
                "Aeq": Aeq,
                "beq": beq,
                "Aineq": Aineq,
                "bineq": bineq,
                "lb": lb,
                "ub": ub,
            }
            import pickle

            pickle.dump(d, f)

    n = c.size
    useStandardForm = False
    if useStandardForm and (not Aineq is None):
        c, Aeq, beq, lb, ub, x0 = convertToStandardFormWithBounds(
            c, Aeq, beq, Aineq, bineq, lb, ub, x0
        )
        Aineq = None

    useColumnPreconditioning = True
    if useColumnPreconditioning:
        # constructing the preconditioning diagonal matrices
        tmp = 0
        if not Aeq is None:
            print("Aeq shape=" + str(Aeq.shape))

            assert scipy.sparse.issparse(Aeq)
            assert Aeq.shape[1] == c.size
            assert Aeq.shape[0] == beq.size
            AeqCopy = Aeq.copy()
            AeqCopy.data = np.abs(AeqCopy.data) ** (2 - alpha)
            SumAeq = np.ones((1, AeqCopy.shape[0])) * AeqCopy
            tmp = tmp + SumAeq
            # AeqT=Aeq.T
        if not Aineq is None:
            print("Aineq shape=" + str(Aineq.shape))
            assert scipy.sparse.issparse(Aineq)
            assert Aineq.shape[1] == c.size
            assert Aineq.shape[0] == bineq.size
            AineqCopy = Aineq.copy()
            AineqCopy.data = np.abs(AineqCopy.data) ** (2 - alpha)
            SumAineq = np.ones((1, AineqCopy.shape[0])) * AineqCopy
            tmp = tmp + SumAineq
            # AineqT=Aineq.T
        if Aeq == None and Aineq == None:
            x = np.zeros_like(lb)
            x[c > 0] = lb[c > 0]
            x[c < 0] = ub[c < 0]
            return x
        tmp[tmp == 0] = 1
        diagT = 1 / tmp[0, :]
        T = scipy.sparse.diags(diagT[None, :], [0]).tocsr()
    else:
        scipy.sparse.eye(len(x))
        diagT = np.ones(x.shape)
    if not Aeq is None:
        AeqCopy = Aeq.copy()
        AeqCopy.data = np.abs(AeqCopy.data) ** (alpha)
        SumAeq = AeqCopy * np.ones((AeqCopy.shape[1]))
        tmp = SumAeq
        tmp[tmp == 0] = 1
        diagSigma_eq = 1 / tmp
        Sigma_eq = scipy.sparse.diags([diagSigma_eq], [0]).tocsr()
        y_eq = np.zeros(Aeq.shape[0])
        del AeqCopy
        del SumAeq
    if not Aineq is None:
        AineqCopy = Aineq.copy()
        AineqCopy.data = np.abs(AineqCopy.data) ** (alpha)
        SumAineq = AineqCopy * np.ones((AineqCopy.shape[1]))
        tmp = SumAineq
        tmp[tmp == 0] = 1
        diagSigma_ineq = 1 / tmp
        Sigma_ineq = scipy.sparse.diags([diagSigma_ineq], [0]).tocsr()
        y_ineq = np.zeros(Aineq.shape[0])
        del AineqCopy
        del SumAineq

    # some cleaning
    del tmp

    # del diagSigma
    # del diagT

    # iterations
    # AeqT=AeqT.tocsc()
    # AineqT=AineqT.tocsc()
    x3 = x

    best_integer_solution_energy = np.inf
    best_integer_solution = None
    for i in range(nb_iter):

        # Update he primal variables
        d = c
        if not Aeq is None:
            if use_vec_sparsity:
                yeq_sparse = scipy.sparse.coo_matrix(y_eq).T
                d = (
                    d + (yeq_sparse * Aeq).toarray().ravel()
                )  # faster when few constraint are activated
            else:
                d = d + y_eq * Aeq
                # d+=y_eq*Aeq# strangley this does not work, give wrong results

        if not Aineq is None:
            if use_vec_sparsity:
                yineq_sparse = scipy.sparse.coo_matrix(y_ineq).T
                d = (
                    d + (yineq_sparse * Aineq).toarray().ravel()
                )  # faster when few constraint are activated
            else:
                d = d + y_ineq * Aineq
                # d+=y_ineq*Aineq

        # x2=x-T*d
        x2 = x - diagT * d
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
        if not Aeq is None:
            if use_vec_sparsity:
                r_eq = (Aeq * x3_sparse).toarray().ravel() - beq
            else:
                r_eq = (Aeq * x3) - beq
        if not Aineq is None:
            if use_vec_sparsity:
                r_ineq = (Aineq * x3_sparse).toarray().ravel() - bineq
            else:
                r_ineq = (Aineq * x3) - bineq

        if i % nb_iter_plot == 0:
            prev_elapsed = elapsed
            elapsed = time.clock() - start
            mean_iter_priod = (elapsed - prev_elapsed) / 10
            if (not max_time is None) and elapsed > max_time:
                break
            energy1 = c.dot(x)

            # x4 is obtained my minimizing with respect to the primal variable while keeping the langrangian coef fix , which give a lower bound on the optimal solution
            # energy2 is the lower bound
            # energy1  is the value of the lagrangian at the curretn (hopefull sadle) point
            # on problem is that the minimmization with respect to the primal variables may actually lead to invintly negative lower bounds...
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
            if not Aeq is None:
                energy1 += y_eq.T.dot(Aeq * x - beq)
                energy2 += y_eq.T.dot(Aeq * x4 - beq)
                max_violated_equality = np.max(np.abs(r_eq))
            if not Aineq is None:
                energy1 += y_ineq.T.dot(Aineq * x - bineq)
                energy2 += y_ineq.T.dot(Aineq * x4 - bineq)
                max_violated_inequality = np.max(r_ineq)
            if force_integer:
                xrounded = np.round(x)
            else:
                xrounded = x
            energy_rounded = c.dot(xrounded)
            if not Aeq is None:
                max_violated_equality_rounded = np.max(np.abs(Aeq * xrounded - beq))
            else:
                max_violated_equality_rounded = 0
            max_violated_inequality = np.max(Aineq * xrounded - bineq)
            if max_violated_equality_rounded == 0 and max_violated_inequality <= 0:
                print(
                    "##########   found feasible solution with energy"
                    + str(energy_rounded)
                )
                if energy_rounded < best_integer_solution_energy:
                    best_integer_solution_energy = energy_rounded
                    best_integer_solution = xrounded

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
                + " x3 has "
                + str(100 * np.mean(x3 == 0))
                + " % of zeros "
                + "diff x3 has "
                + str(100 * np.mean(diff_x3 == 0))
                + " % of zeros "
                + "mean_iter_period="
                + str(mean_iter_priod)
            )
            #'y_eq has '+str(100 * np.mean(y_eq==0))+' % of zeros '+\
            #    'y_ineq has '+str(100 * np.mean(y_ineq==0))+' % of zeros '+\

            if not callbackFunc is None:

                callbackFunc(
                    i,
                    x,
                    energy1,
                    energy2,
                    elapsed,
                    max_violated_equality,
                    max_violated_inequality,
                )

        # Update the dual variables

        if not Aeq is None:
            y_eq = y_eq + Sigma_eq * r_eq
            # y_eq=y_eq+diagSigma_eq*r_eq
            # y_eq+=diagSigma_eq*r_eq

        if not Aineq is None:
            y_ineq = y_ineq + Sigma_ineq * r_ineq
            # y_ineq+=diagSigma_ineq*r_ineq
            np.maximum(y_ineq, 0, y_ineq)
            # y_ineq=np.maximum(y_ineq, 0)

    if not best_integer_solution is None:
        best_integer_solution = best_integer_solution[:n]
    return x[:n], best_integer_solution
