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
"""Model that implements various utils functions used in LP solvers."""

import time

import numpy as np

import scipy.sparse


class Chrono:
    """Small class to compute durations."""

    def __init__(self):
        pass

    def tic(self):
        self.start = time.clock()

    def toc(self):
        return time.clock() - self.start


class CheckDecrease:
    """Class to help checking decrease of a value."""

    def __init__(self, val=None, tol=1e-10):
        self.val = val
        self.tol = tol

    def set_value(self, val):
        self.val = val

    def add_value(self, val):
        assert self.val >= val - self.tol
        self.val = val


def convert_to_py_sparse_format(a):
    # check symmetric
    import spmatrix
    assert (a - a.T).nnz == 0
    l_mat = spmatrix.ll_mat_sym(a.shape[0], a.nnz)
    a_coo = scipy.sparse.triu(a).tocoo()
    l_mat.put(a_coo.data, a_coo.row.astype(int), a_coo.col.astype(int))

    return l_mat


class CholeskyOrLu:
    """Class to wrap linear solvers."""

    def __init__(self, m, method):
        if method == "scipySparseLu":
            self.LU = scipy.sparse.linalg.splu(m.tocsc())
            self.solve = self.LU.solve
        elif method == "scikitsCholesky":
            import scikits.sparse
            self.LU = scikits.sparse.cholmod.cholesky(m.tocsc())
            self.solve = self.LU.solve_A


def convert_to_standard_form_with_bounds(c, a_eq, beq, a_ineq, b_lower, b_upper, lb, ub, x0):

    if a_ineq is not None:
        ni = a_ineq.shape[0]
        # need to convert in standard form by adding an auxiliary variables for each inequality
        if a_eq is not None:
            a_eq2 = scipy.sparse.vstack(
                (
                    scipy.sparse.hstack(
                        (a_eq, scipy.sparse.csc_matrix((a_eq.shape[0], ni)))
                    ),
                    scipy.sparse.hstack((a_ineq, -scipy.sparse.eye(ni, ni))),
                )
            ).tocsr()
            a_eq2.__dict__["blocks"] = a_eq.blocks + [
                (b[0] + a_eq.shape[0], b[1] + a_eq.shape[0]) for b in a_ineq.blocks
            ]
            b_eq2 = np.hstack((beq, np.zeros((ni))))
        else:

            a_eq2 = scipy.sparse.hstack((a_ineq, -scipy.sparse.eye(ni, ni))).tocsr()
            a_eq2.__dict__["blocks"] = a_ineq.blocks
            b_eq2 = np.zeros((ni))

        if b_lower is None:
            b_lower = np.empty(a_ineq.shape[0])
            b_lower.fill(-np.inf)

        if b_upper is None:
            b_upper = np.empty(a_ineq.shape[0])
            b_upper.fill(np.inf)

        lb = np.hstack((lb, b_lower))
        ub = np.hstack((ub, b_upper))
        epsilon0 = a_ineq * x0
        x0 = np.hstack((x0, epsilon0))
        c = np.hstack((c, np.zeros(ni)))
    return c, a_eq2, b_eq2, lb, ub, x0


def convert_to_one_sided_inequality_system(a_ineq, b_lower, b_upper):
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
    return a_ineq, b_ineq


def check_constraints(i, x_r, mask, a_csr, a_csc, b_lower, b_upper):
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


class SolutionStat:
    """Class that compute statistics of solution of an LP problem."""

    def __init__(self, c, a_eq_csc, beq, a_ineq_csc, b_ineq, callback_func):
        self.c = c
        self.a_eq = a_eq_csc
        self.beq = beq
        self.a_ineq = a_ineq_csc
        self.b_ineq = b_ineq
        self.best_integer_solution_energy = np.inf
        self.best_integer_solution = None
        self.iprev = 0
        self.callback_func = callback_func

    def start_timer(self):
        self.start = time.clock()
        self.elapsed = self.start

    def evaluate(self, x, i):

        self.prev_elapsed = self.elapsed
        elapsed = time.clock() - self.start
        nb_iter_since_last_call = i - self.self.iprev
        mean_iter_period = (
            elapsed - self.prev_elapsed) / nb_iter_since_last_call

        energy1 = self.c.dot(x)
        max_violated_equality = 0
        max_violated_inequality = 0
        r_eq = (self.a_eq * x) - self.beq
        r_ineq = (self.a_ineq * x) - self.b_ineq
        if self.a_eq is not None:
            max_violated_equality = np.max(np.abs(r_eq))
        if self.a_ineq is not None:
            max_violated_inequality = np.max(r_ineq)

        x_rounded = np.round(x)
        energy_rounded = self.c.dot(x_rounded)
        nb_violated_equality_rounded = np.sum(np.abs(self.a_eq * x_rounded - self.beq))
        nb_violated_inequality_rounded = np.sum(
            np.maximum(self.a_ineq * x_rounded - self.b_ineq, 0))

        if nb_violated_equality_rounded == 0 and nb_violated_inequality_rounded == 0:
            print(
                "##########   found feasible solution with energy" + str(energy_rounded)
            )
            if energy_rounded < self.best_integer_solution_energy:
                self.best_integer_solution_energy = energy_rounded
                self.best_integer_solution = x_rounded

        print(
            "iter"
            + str(i)
            + ": energy1= "
            + str(energy1)
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
        self.iprev = i


def save_arguments(filename):
    """Return tuple containing dictionary of calling function's
    named arguments and a list of calling function's unnamed
    positional arguments.
    """
    from inspect import getargvalues, stack
    import inspect

    pos_name, kw_name, args = getargvalues(stack()[1][0])[-3:]
    pos_args = args.pop(pos_name, [])
    args.update(args.pop(kw_name, []))
    caller = inspect.currentframe().f_back
    func_name = caller.f_code.co_name

    module = caller.f_globals["__name__"]
    import pickle

    d = {"module": module, "function_name": func_name, "args": args, "posargs": pos_args}
    with open(filename, "wb") as f:
        pickle.dump(d, f)


def precondition_constraints(a, b, b2=None, alpha=2):
    # alpha=2
    a_copy = a.copy()
    a_copy.data = np.abs(a_copy.data) ** (alpha)
    sum_a = a_copy * np.ones((a_copy.shape[1]))
    tmp = (sum_a) ** (1.0 / alpha)
    tmp[tmp == 0] = 1
    diag_sigm_a = 1 / tmp
    sigma = scipy.sparse.diags([diag_sigm_a], [0]).tocsr()
    a_p = sigma * a
    a_p.__dict__["blocks"] = a.blocks
    if b is not None:
        bp = sigma * b
    else:
        bp = None
    if b2 is None:
        return a_p, bp
    else:
        return a_p, bp, sigma * b2


def precondition_lp_right(c, a_eq, beq, lb, ub, x0, alpha=2):
    # alpha=2
    a_eq_copy = a_eq.copy()
    a_eq_copy.data = np.abs(a_eq_copy.data) ** (alpha)
    sum_a = np.ones((a_eq_copy.shape[0])) * a_eq_copy
    tmp = (sum_a) ** (1.0 / alpha)
    tmp[tmp == 0] = 1
    diag_r = 1 / tmp
    r = scipy.sparse.diags([diag_r], [0]).tocsr()

    a_eq2 = a_eq * r
    b_eq2 = beq
    lb2 = tmp * lb
    ub2 = tmp * ub
    x02 = tmp * x0
    c2 = c * r
    a_eq2.__dict__["blocks"] = a_eq.blocks

    return r, c2, a_eq2, b_eq2, lb2, ub2, x02
