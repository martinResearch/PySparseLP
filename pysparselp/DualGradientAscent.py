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
import scipy.sparse
import scipy.ndimage


def exactDualLineSearch(direction, A, b, c_bar, upperbounds, lowerbounds):

    assert isinstance(direction, scipy.sparse.csr.csr_matrix)
    dA = direction * A
    alphas = -c_bar[dA.indices] / dA.data
    order = np.argsort(alphas)
    dAU = dA.data * upperbounds[dA.indices]
    dAL = dA.data * lowerbounds[dA.indices]
    tmp1 = np.minimum(dAU[order], dAL[order])
    tmp2 = np.maximum(dAU[order], dAL[order])
    tmp3 = np.cumsum(tmp2[::-1])[::-1]
    tmp4 = np.cumsum(tmp1)
    derivatives = -(direction.dot(b)) * np.ones(alphas.size + 1)
    derivatives[:-1] += tmp3
    derivatives[1:] += tmp4

    # tmp=np.abs(Ai.data[order])*(LP2.lowerbounds[Ai.indices[order]]-LP2.upperbounds[Ai.indices[order]])
    # derivatives= -LP2.Bequalities[i]-np.sum(AiL[Ai.data>0])-np.sum(AiU[Ai.data<0])\
    # +np.hstack(([0],np.cumsum(tmp)))

    k = np.searchsorted(-derivatives, 0)
    if derivatives[k] == 0 and k < len(order):
        t = np.random.rand()
        alpha_optim = (
            t * alphas[order[k]] + (1 - t) * alphas[order[k - 1]]
        )  # maybe courld draw and random valu in the interval ?
    else:

        alpha_optim = alphas[order[k - 1]]
    return alpha_optim


def DualGradientAscent(
    x,
    LP,
    nbmaxiter=1000,
    callbackFunc=None,
    y_eq=None,
    y_ineq=None,
    max_time=None,
    nb_iter_plot=1,
):
    """gradient ascent in the dual
	"""
    start = time.clock()
    # convert to slack form (augmented form)
    LP2 = copy.deepcopy(LP)
    LP = None
    # LP2.convertToSlackForm()
    assert (LP2.B_lower is None) or np.max(LP2.B_lower) == -np.inf
    # y_ineq=None
    # LP2.convertTo
    # LP2.convertToOnesideInequalitySystem()
    # LP2.upperbounds=np.minimum(10000,LP2.upperbounds)
    # LP2.lowerbounds=np.maximum(-10000,LP2.lowerbounds)
    # y0=None
    if y_eq is None:
        y_eq = np.zeros(LP2.Aequalities.shape[0])
        y_eq = -np.random.rand(y_eq.size)
    else:
        y_eq = y_eq.copy()
    # y_ineq=None
    if y_ineq is None:
        if not (LP2.Ainequalities is None):
            y_ineq = np.zeros(LP2.Ainequalities.shape[0])
            y_ineq = np.abs(np.random.rand(y_ineq.size))
    else:
        y_ineq = y_ineq.copy()
    # assert (LP2.B_lower is None)

    def getOptimX(y_eq, y_ineq):
        c_bar = LP2.costsvector.copy()
        if not LP2.Aequalities is None:
            c_bar += y_eq * LP2.Aequalities
        if not LP2.Ainequalities is None:
            c_bar += y_ineq * LP2.Ainequalities
        x = np.zeros(LP2.costsvector.size)
        x[c_bar > 0] = LP2.lowerbounds[c_bar > 0]
        x[c_bar < 0] = LP2.upperbounds[c_bar < 0]
        x[c_bar == 0] = 0.5 * (LP2.lowerbounds + LP2.upperbounds)[c_bar == 0]
        # t=np.random.rand(np.sum(c_bar==0))
        # x[c_bar==0]=t*LP2.lowerbounds[c_bar==0]+(1-t)*LP2.upperbounds[c_bar==0]

        return c_bar, x

    def eval(y_eq, y_ineq):
        c_bar, x = getOptimX(y_eq, y_ineq)

        # E=-y_eq.dot(LP2.Bequalities)-y_ineq.dot(LP2.B_upper)+np.sum(x*c_bar)
        # LP2.costsvector.dot(x)+y_ineq.dot(LP2.Ainequalities*x-LP2.B_upper)
        E = np.sum(
            np.minimum(c_bar * LP2.upperbounds, c_bar * LP2.lowerbounds)[c_bar != 0]
        )
        if not LP2.Aequalities is None:
            E -= y_eq.dot(LP2.Bequalities)
        if not LP2.Ainequalities is None:
            E -= y_ineq.dot(LP2.B_upper)
        return E

    # x[c_bar==0]=0.5

    # alpha_i= vector containing the step lenghts that lead to a sign change on any of the gradient component
    # when incrementing y[i]
    #
    print("iter %d energy %f" % (0, eval(y_eq, y_ineq)))

    prevE = eval(y_eq, y_ineq)
    if prevE == -np.inf:
        print("initial dual point not feasible, you could bound all variables")
        c_bar, x = getOptimX(y_eq, y_ineq)
        return x, y_eq, y_ineq
    for iter in range(nbmaxiter):
        c_bar, x = getOptimX(y_eq, y_ineq)
        if not LP2.Ainequalities is None:
            y_ineq_prev = y_ineq.copy()
            max_violation = np.max(LP2.Ainequalities * x - LP2.B_upper)
            sum_violation = np.sum(np.maximum(LP2.Ainequalities * x - LP2.B_upper, 0))
            np.sum(np.maximum(LP2.Ainequalities * x - LP2.B_upper, 0))
            if (iter % nb_iter_plot) == 0:
                print(
                    "iter %d energy %f max violation %f sum_violation %f"
                    % (iter, prevE, max_violation, sum_violation)
                )

            grad_y_ineq = LP2.Ainequalities * x - LP2.B_upper

            grad_y_ineq[y_ineq_prev <= 0] = np.maximum(
                grad_y_ineq[y_ineq_prev <= 0], 0
            )  # not sure it is correct to do that here
            if np.sum(grad_y_ineq < 0) > 0:

                grad_y_ineq_sparse = scipy.sparse.csr.csr_matrix(grad_y_ineq)
                coef_length_ineq = exactDualLineSearch(
                    grad_y_ineq_sparse,
                    LP2.Ainequalities,
                    LP2.B_upper,
                    c_bar,
                    LP2.upperbounds,
                    LP2.lowerbounds,
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
                # vals.append(eval(y_eq,y2))
                # plt.plot(alphasgrid,vals,'.')

                # coef_length=0.001/(iter+2000000)
                # coef_length=min(0.01/(iter+200000),maxstep)
                y_ineq = y_ineq_prev + coef_length_ineq * grad_y_ineq
                # assert(np.min(y_ineq)>=-1e-8)
                y_ineq = np.maximum(y_ineq, 0)

        if not LP2.Aequalities is None and LP2.Aequalities.shape[0] > 0:

            y_eq_prev = y_eq.copy()
            max_violation = np.max(np.abs(LP2.Aequalities * x - LP2.Bequalities))
            sum_violation = np.sum(np.abs(LP2.Aequalities * x - LP2.Bequalities))
            if (iter % nb_iter_plot) == 0:
                print(
                    "iter %d energy %f max violation %f sum_violation %f"
                    % (iter, prevE, max_violation, sum_violation)
                )

            grad_y_eq = LP2.Aequalities * x - LP2.Bequalities
            if np.any(grad_y_eq):
                grad_y_eq_sparse = scipy.sparse.csr.csr_matrix(grad_y_eq)
                coef_length_eq = exactDualLineSearch(
                    grad_y_eq_sparse,
                    LP2.Aequalities,
                    LP2.Bequalities,
                    c_bar,
                    LP2.upperbounds,
                    LP2.lowerbounds,
                )
                # y_ineq_prev+coef_length*grad_y>0
                assert coef_length_eq >= 0

                y_eq = y_eq_prev + coef_length_eq * grad_y_eq

        # while True:
        # y_ineq=y_ineq_prev+coef_length*grad_y
        # newE=eval(y_eq,y_ineq)
        # if newE< prevE:
        # coef_length=coef_length*0.5
        # print 'reducing step lenght'
        # else:
        # coef_length=coef_length*1.5
        # break
        newE = eval(y_eq, y_ineq)
        prevE = newE
        elapsed = time.clock() - start
        if not callbackFunc is None and iter % 100 == 0:
            callbackFunc(iter, x, 0, 0, elapsed, 0, 0)

        if (not max_time is None) and elapsed > max_time:
            break

    print("done")
    return x, y_eq, y_ineq
