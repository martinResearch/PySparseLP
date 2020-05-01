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
"""Implementation of an LP solver using the alternating direction method of multipliers (ADMM) method."""
import time

import numpy as np

import scipy.ndimage
import scipy.sparse

from .conjugateGradientLinearSolver import conjgrad
from .gaussSiedel import GaussSeidel
from .gaussSiedel import boundedGaussSeidelClass
from .tools import (
    Chrono,
    convert_to_py_sparse_format,
    convert_to_standard_form_with_bounds,
    precondition_constraints,
)

# import  scikits.sparse.cholmod
# @profile


def lp_admm(
    c,
    a_eq,
    beq,
    a_ineq,
    b_lower,
    b_upper,
    lb,
    ub,
    x0=None,
    gamma_eq=2,
    gamma_ineq=3,
    nb_iter=100,
    callback_func=None,
    max_duration=None,
    use_preconditioning=True,
    nb_iter_plot=10,
):
    # simple ADMM method with an approximate resolution of a quadratic subproblem using conjugate gradient
    use_lu = False
    use_cholesky = False
    use_amg = False
    use_cg = False
    use_bounded_gauss_siedel = True
    use_unbounded_gauss_siedel = False

    n = c.size
    if x0 is None:
        x0 = np.zeros(c.size)
    if a_eq is not None:
        a_eq, beq = precondition_constraints(a_eq, beq, alpha=2)
    if (
        a_ineq is not None
    ):  # it seem important to do this preconditioning before converting to standard form
        a_ineq, b_lower, b_upper = precondition_constraints(
            a_ineq, b_lower, b_upper, alpha=2
        )
    c, a_eq, beq, lb, ub, x0 = convert_to_standard_form_with_bounds(
        c, a_eq, beq, a_ineq, b_lower, b_upper, lb, ub, x0
    )
    x = x0

    # trying some preconditioning
    if use_preconditioning:
        a_eq, beq = precondition_constraints(a_eq, beq, alpha=2)

    a_t_a = a_eq.T * a_eq
    # AAt=a_eq*a_eq.T
    a_t_b = a_eq.T * beq
    identity = scipy.sparse.eye(x.size, x.size)

    xp = np.maximum(x, 0)

    m = gamma_eq * a_t_a + gamma_ineq * identity
    m = m.tocsr()
    lambda_eq = np.zeros(a_eq.shape[0])
    lambda_ineq = np.zeros(x.shape)
    if use_lu:
        lu_m = scipy.sparse.linalg.splu(m)
        # luM = scipy.sparse.linalg.spilu(M,drop_tol=0.01)
    elif use_cholesky:
        import scikits.sparse

        ch = Chrono()
        ch.tic()
        chol = scikits.sparse.cholmod.cholesky(m.tocsc())
        print("cholesky factorization took " + str(ch.toc()) + " seconds")
        print(
            "the sparsity ratio between the cholesky decomposition of M and M is "
            + str(chol.L().nnz / float(m.nnz))
        )

    elif use_amg:
        import pyamg

        m_amg = pyamg.ruge_stuben_solver(m)

    def energy(x, xp, lambda_eq, lambda_ineq):
        en = (
            c.dot(x)
            + 0.5 * gamma_eq * np.sum((a_eq * x - beq) ** 2)
            + 0.5 * gamma_ineq * np.sum((x - xp) ** 2)
            + lambda_eq.dot(a_eq * x - beq)
            + lambda_ineq.dot(x - xp)
        )
        return en

    i = 0
    nb_cg_iter = 1
    speed = np.zeros(x.shape)

    order = np.arange(x.size).astype(np.uint32)
    bs = boundedGaussSeidelClass(m)
    alpha = 1.4
    start = time.clock()
    elapsed = start
    while i <= nb_iter / nb_cg_iter:
        # solve the penalized problem with respect to x
        # c +gamma_eq*(a_t_a x-a_t_b) + gamma_ineq*(x -xp)+lambda_eq*a_eq+lambda_ineq
        # M*x=-c+a_t_b+gamma_ineq*xp-lambdas-lambda_eq*a_eq

        y = -c + gamma_eq * a_t_b + gamma_ineq * xp - lambda_eq * a_eq - lambda_ineq
        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_eq,lambda_ineq))
        if use_lu:
            x = lu_m.solve(y)
        elif use_cholesky:
            x = chol.solve_A(y)
        elif use_bounded_gauss_siedel:
            xprev = x.copy()

            # x=xprev+1*speed	# maybe could do a line search along that direction ?
            # if i%2==0:
            # order=np.arange(x.size).astype(np.uint32)
            # else:
            # order=np.arange(x.size-1,-1,-1).astype(np.uint32)
            bs.solve(y, lb, ub, x, maxiter=nb_cg_iter, w=1, order=order)
            speed = x - xprev
        elif use_unbounded_gauss_siedel:
            xprev = x.copy()
            # x=xprev+1*speed	# predict the minimum , can yield to some speedup
            if (
                False
            ):  # optimal sep along the direction given by the last two iterates, does not seem to improve much speed
                direction = speed
                t = -direction.dot(m * xprev - y)
                print(t)
                if abs(t) > 0:
                    step_length = t / (direction.dot(m * direction))
                    x = xprev + step_length * direction
            else:
                pass
                # x=xprev+0.8*speed
            GaussSeidel(m, y, x, maxiter=nb_cg_iter, w=1.0)
            speed = x - xprev
            x = alpha * x + (1 - alpha) * xp
        elif use_cg:
            # x1,r=scipy.sparse.linalg.cgs(M, y,  maxiter=2,x0=x)
            xprev = x.copy()

            if (
                True
            ):  # optimal sep along the direction given by the last two iterates, doe not seem to improve things in term of number of iteration , and slow down iterations...
                # maybe use next step as a conjugate step would help ?
                direction = speed
                t = -direction.dot(m * x - y)
                if abs(t) > 0:
                    step_length = t / (direction.dot(m * direction))
                    x = x + step_length * direction
            else:
                # x=xprev+1*speed			# does not work with cg, explode
                pass
            # start conjugate gradient from there (could use previous direction ? )
            x = conjgrad(m, y, maxiter=nb_cg_iter, x0=x)
            speed = x - xprev
            x = alpha * x + (1 - alpha) * xp
        elif use_amg:
            # xprev=x.copy()
            # x=xprev+1*speed
            x = m_amg.solve(y, x0=x, tol=1e-3)
            # speed=x-xprev
            x = alpha * x + (1 - alpha) * xp  # over relaxation

        else:
            print("unkown method")
            raise

        if i % nb_iter_plot == 0:

            elapsed = time.clock() - start
            if max_duration is not None and elapsed > max_duration:
                break
            energy1 = energy(x, xp, lambda_eq, lambda_ineq)
            energy2 = energy1
            r = a_eq * x - beq
            max_violated_equality = np.max(np.abs(r))
            max_violated_inequality = max(0, -np.min(x))

            print(
                f"iter {i} elapsed={elapsed:1.2f}: energy1={energy1} energy2={energy2}"
                f"max violated inequality:{max_violated_inequality:1.3e}"
                f"max violated max_violated_equality:{max_violated_equality:1.3e}"
            )
            if callback_func is not None:
                callback_func(
                    i,
                    x[0:n],
                    energy1,
                    energy2,
                    elapsed,
                    max_violated_equality,
                    max_violated_inequality,
                )

        # solve the penalized problem with respect to xp
        # -gamma_ineq*(x-xp)-lambda_ineq=0
        if not (use_bounded_gauss_siedel):
            xp = x.copy() + lambda_ineq / gamma_ineq
            xp = np.maximum(xp, lb)
            xp = np.minimum(xp, ub)
            lambda_ineq = lambda_ineq + gamma_ineq * (x - xp)
            # print np.max(np.abs(lambda_ineq))
        else:
            xp = x
        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_eq,lambda_ineq))
        lambda_eq = lambda_eq + gamma_eq * (
            a_eq * x - beq
        )  # could use heavy ball instead of gradient step ?

        # could try to update the penalty ?
        # gamma_ineq=gamma_ineq+
        # M=gamma_eq*a_t_a+gamma_ineq*Id
        i += 1
    return x[0:n]


def lp_admm2(
    c,
    a_eq,
    beq,
    a_ineq,
    b_lower,
    b_upper,
    lb,
    ub,
    x0=None,
    gamma_ineq=0.7,
    nb_iter=100,
    callback_func=None,
    max_duration=None,
    use_preconditioning=False,
    nb_iter_plot=10,
):
    # simple ADMM method with an approximate resolution of a quadratic subproblem using conjugate gradient
    # inspired by Boyd's paper on ADMM
    # Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
    # the difference with admm_solver is that the linear equality constraints a_eq*x=beq are enforced during the resolution
    # of the subproblem instead of beeing enforced through multipliers
    use_lu = True
    use_amg = False
    use_cholesky = False
    use_cholesky2 = False

    # relaxation parameter should be in [0,2] , 1.95 seems to be often a good choice
    alpha = 1.95

    start = time.clock()
    elapsed = start
    n = c.size
    if x0 is None:
        x0 = np.zeros(c.size)

    if use_preconditioning:
        if a_eq is not None:
            a_eq, beq = precondition_constraints(a_eq, beq, alpha=2)
        if (
            a_ineq is not None
        ):  # it seem important to do this preconditioning before converting to standard form
            a_ineq, b_lower, b_upper = precondition_constraints(
                a_ineq, b_lower, b_upper, alpha=2
            )

    c, a_eq, beq, lb, ub, x0 = convert_to_standard_form_with_bounds(
        c, a_eq, beq, a_ineq, b_lower, b_upper, lb, ub, x0
    )
    x = x0

    xp = x.copy()
    xp = np.maximum(xp, lb)
    xp = np.minimum(xp, ub)
    ch = Chrono()
    # trying some preconditioning
    if use_preconditioning:
        a_eq, beq = precondition_constraints(a_eq, beq, alpha=2)

    m = scipy.sparse.vstack(
        (
            scipy.sparse.hstack(
                (gamma_ineq * scipy.sparse.eye(a_eq.shape[1], a_eq.shape[1]), a_eq.T)
            ),
            scipy.sparse.hstack(
                (a_eq, scipy.sparse.csc_matrix((a_eq.shape[0], a_eq.shape[0])))
            ),
        )
    ).tocsr()
    if use_lu:
        lu_m = scipy.sparse.linalg.splu(m.tocsc())
        nb_cg_iter = 1
    elif use_cholesky:
        import scikits.sparse

        ch.tic()
        # not that it will work only if M is positive definite which nto garantied the way it is constructed
        # unfortunately i'm not able to catch the error to fall back on LU decomposition if
        # cholesky fails because the matrix is not positive definite
        chol = scikits.sparse.cholmod.cholesky(
            m.tocsc(), mode="simplicial"
        )  # pip install scikit-sparse, but difficult to compile in windows

        print("cholesky factorization took " + str(ch.toc()) + " seconds")
        print(
            "the sparsity ratio between the cholesky decomposition of M and M is "
            + str(chol.L().nnz / float(m.nnz))
        )
        nb_cg_iter = 1
    elif use_cholesky2:
        import scikits.umfpack  # pipinstall scikit-umfpack

        print("using UMFPACK_STRATEGY_SYMMETRIC through PySparse")
        ch.tic()
        m2 = convert_to_py_sparse_format(m)
        print("conversion :" + str(ch.toc()))
        ch.tic()
        lu_umfpack = scikits.umfpack.factorize(
            m2, strategy="UMFPACK_STRATEGY_SYMMETRIC"
        )
        print("nnz per line :" + str(lu_umfpack.nnz / float(m2.shape[0])))
        print("factorization :" + str(ch.toc()))
        nb_cg_iter = 1

    elif use_amg:
        # Mamg=pyamg.smoothed_aggregation_solver(M.tocsc())
        # Mamg=pyamg.rootnode_solver(M.tocsc())
        # Mamg=pyamg.
        import pyamg  # pip install pyamg

        m_amg = pyamg.ruge_stuben_solver(
            m.tocsc(), strength=None
        )  # sometimes seems to yield infinite values
        # I=scipy.sparse.eye(1000)
        # Mamg=pyamg.ruge_stuben_solver(I.tocsc())

        for l in range(len(m_amg.levels)):
            print("checking level " + str(l))
            assert np.isfinite(m_amg.levels[l].A.data).all()
        nb_cg_iter = 1
    else:
        nb_cg_iter = 100
    lambda_ineq = np.zeros(x.shape)

    def energy(x, xp, lambda_ineq):
        en = (
            c.dot(x)
            + 0.5 * gamma_ineq * np.sum((x - xp) ** 2)
            + lambda_ineq.dot(x - xp)
        )
        return en

    niter = 0
    xv = np.hstack((x, np.zeros(beq.shape)))

    while niter <= nb_iter / nb_cg_iter:
        # solve the penalized problem with respect to x
        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))

        y = np.hstack((-c + gamma_ineq * xp - lambda_ineq, beq))
        if use_lu:
            xv = lu_m.solve(y)
        elif use_cholesky:
            xv = chol.solve_A(y)
        elif use_cholesky2:

            lu_umfpack.solve(y, xv)

        elif use_amg:
            xv = m_amg.solve(y, x0=xv, tol=1e-12)
            if np.linalg.norm(m * xv - y) > 1e-5:
                raise

        else:
            xv = conjgrad(m, y, maxiter=nb_cg_iter, x0=xv)
        x = xv[: x.shape[0]]
        x = alpha * x + (1 - alpha) * xp

        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))
        # solve the penalized problem with respect to xp
        # -gamma_ineq*(x-xp)-lambda_ineq=0
        xp = x.copy() + lambda_ineq / gamma_ineq
        xp = np.maximum(xp, lb)
        xp = np.minimum(xp, ub)
        if niter % nb_iter_plot == 0:
            elapsed = time.clock() - start
            if not (max_duration is None) and elapsed > max_duration:
                break
            energy1 = energy(x, xp, lambda_ineq)
            energy2 = energy1

            max_violated_equality = 0
            max_violated_inequality = 0
            print(
                "iter"
                + str(niter)
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
                    niter,
                    x[0:n],
                    energy1,
                    energy2,
                    elapsed,
                    max_violated_equality,
                    max_violated_inequality,
                )

        # print 'iter'+str(i)+' '+str(L(x, xp,lambda_ineq))
        lambda_ineq = lambda_ineq + gamma_ineq * (x - xp)
        niter += 1
    return x[0:n]
