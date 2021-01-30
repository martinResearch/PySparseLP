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
"""Module that implement the class SparseLP to help modeling the LP problem."""

import copy
import time

import numpy as np

import scipy.ndimage
import scipy.optimize
import scipy.sparse


from .ADMM import lp_admm, lp_admm2
from .ADMMBlocks import lp_admm_block_decomposition
from .ChambollePockLineSearch import chambolle_pock_linesearch
from .ChambollePockPPD import chambolle_pock_ppd
from .DualCoordinateAscent import dual_coordinate_ascent
from .DualGradientAscent import dual_gradient_ascent
from .MehrotraPDIP import mpc_sol


solving_methods = (
    "mehrotra",
    "chambolle_pock_ppd",
    "chambolle_pock_linesearch",
    "admm",
    "admm2",
    "admm_blocks",
    "scipy_simplex",
    "scipy_revised_simplex",
    "scipy_interior_point",
    "dual_coordinate_ascent",
    "dual_gradient_ascent",
)

try:
    import cvxpy

    solving_methods += ("ECOS", "SCS")

except Exception:
    print("could not import cvxpy, the ECOS and SCS solvers will not be available")

try:
    import osqp

    solving_methods += ("osqp",)

except Exception:
    print("could not import osqp. The osqp solver will not be available")


def csr_matrix_append_row(a, n, cols, vals):
    a.blocks.append((a.shape[0], a.shape[0]))
    a._shape = (a.shape[0] + 1, n)
    a.indices = np.append(a.indices, cols.astype(a.indices.dtype))
    a.data = np.append(a.data, vals.astype(a.data.dtype))
    a.indptr = np.append(a.indptr, np.int32(a.indptr[-1] + cols.size))
    assert a.data.size == a.indices.size
    assert a.indptr.size == a.shape[0] + 1
    assert a.indptr[-1] == a.data.size


def check_csr_matrix(a):
    assert np.max(a.indices) < a.shape[1]
    assert len(a.data) == len(a.indices)
    assert len(a.indptr) == a.shape[0] + 1
    assert np.all(np.diff(a.indptr) >= 0)


def csr_matrix_append_rows(a, b):
    # A._shape=-A.shape[0],B._shape[1])
    a.blocks.append((a.shape[0], a.shape[0] + b.shape[0] - 1))
    a._shape = (a.shape[0] + b.shape[0], max(a.shape[1], b.shape[1]))
    a.indices = np.append(a.indices, b.indices)
    a.data = np.append(a.data, b.data)
    a.indptr = np.append(a.indptr[:-1], a.indptr[-1] + b.indptr)

    assert np.max(a.indices) < a.shape[1]
    assert a.data.size == a.indices.size
    assert a.indptr.size == a.shape[0] + 1
    assert a.indptr[-1] == a.data.size


def empty_csr_matrix():
    a = scipy.sparse.csr_matrix((1, 1), dtype=np.float)
    # trick , because it would not let me create and empty matrix
    a._shape = (0 * a._shape[0], 0 * a._shape[1])
    a.indptr = a.indptr[-1:]
    return a


def unique_rows(data, prec=5):
    import numpy as np

    d_r = np.fix(data * 10 ** prec) / 10 ** prec + 0.0
    b = np.ascontiguousarray(d_r).view(
        np.dtype((np.void, d_r.dtype.itemsize * d_r.shape[1]))
    )
    _, ia = np.unique(b, return_index=True)
    _, ic = np.unique(b, return_inverse=True)
    return np.unique(b).view(d_r.dtype).reshape(-1, d_r.shape[1]), ia, ic


def crd_matrix(cols, vals, broadcast=True):
    """Construct a compressed sparse row matrix with constant number of non zero value per row.

    the matrix is in the form m[i,cols[i,j]]=val[i,j]
    By default the array cols and vals are broadcasted in order to make it easier to use
    """
    assert np.ndim(cols) == 2
    assert np.ndim(vals) == 2

    if not (np.all(np.diff(np.sort(cols, axis=1), axis=1) > 0)):
        unvalid = np.nonzero(
            np.any(np.diff(np.sort(cols, axis=1), axis=1) == 0, axis=1)
        )[0]
        error_message = (
            f"you have twice the same variable in {len(unvalid)} constraint"
            + ["", "s"][len(unvalid) > 0]
            + ":\n"
            + str(unvalid)
        )
        raise (error_message)

    if broadcast:
        cols, vals = np.broadcast_arrays(cols, vals)
    assert np.ndim(cols) == 2
    assert np.all(vals.shape == cols.shape)

    vals_flat = vals.ravel()
    cols_flat = cols.ravel()
    keep = ~(vals == 0)
    vals_flat = vals_flat[keep.ravel()]
    cols_flat = cols_flat[keep.ravel()]
    iptr = np.hstack(([0], np.cumsum(np.sum(keep, axis=1))))
    return scipy.sparse.csr_matrix((vals_flat, cols_flat, iptr))


class SparseLP:
    """Class to help modeling the LP problem."""

    def __init__(self):
        # start writing the linear program

        self.nb_variables = 0
        self.variables_dict = dict()
        self.upper_bounds = np.empty((0), dtype=np.float)
        self.lower_bounds = np.empty((0), dtype=np.float)
        self.costsvector = np.empty((0), dtype=np.float)
        self.is_integer = np.empty((0), dtype=np.bool)
        self.a_inequalities = empty_csr_matrix()
        self.a_inequalities.__dict__["blocks"] = []
        self.b_lower = np.empty((0), dtype=np.float)
        self.b_upper = np.empty((0), dtype=np.float)
        self.a_equalities = empty_csr_matrix()
        self.b_equalities = np.empty((0), dtype=np.float)
        self.a_equalities.__dict__["blocks"] = []
        self.solver = "chambolle_pock"
        self.equalityConstraintNames = []
        self.inequalityConstraintNames = []
        self.solution = None
        self.dual_solution = None

    def max_constraint_violation(self, solution):
        types, lb, ub = self.get_variables_bounds()
        max_v = 0
        max_v = max(max_v, np.max(lb - solution))
        max_v = max(max_v, np.max(solution - ub))
        if self.a_equalities.shape[0] > 0:
            max_v = max(
                max_v, np.max(np.abs(self.a_equalities * solution - self.b_equalities))
            )
        if self.a_inequalities.shape[0] > 0:
            if self.b_upper is not None:
                max_v = max(
                    max_v, np.max(self.a_inequalities * solution - self.b_upper)
                )
            if self.b_lower is not None:
                max_v = max(
                    max_v, np.max(self.b_lower - self.a_inequalities * solution)
                )
        return max_v

    def check_solution(self, solution, tol=1e-6):
        assert solution.ndim == 1
        types, lb, ub = self.get_variables_bounds()
        valid = True
        if lb is not None:
            valid = valid & (np.max(lb - solution) < tol)
        if ub is not None:
            valid = valid & (np.max(solution - ub) < tol)
        if (self.a_equalities is not None) and self.a_equalities.shape[0] > 0:
            valid = valid & (
                np.max(np.abs(self.a_equalities * solution - self.b_equalities)) < tol
            )
        if (self.a_inequalities is not None) and self.a_inequalities.shape[0] > 0:
            if self.b_upper is not None:
                valid = valid & (
                    np.max(self.a_inequalities * solution - self.b_upper) < tol
                )
            if self.b_lower is not None:
                valid = valid & (
                    np.max(self.b_lower - self.a_inequalities * solution) < tol
                )
        return valid

    def start_constraint_name(self, name):
        if not (name is None or name == ""):
            self.lastNameStart = name
            self.lastNameEqualityStart = self.nb_equality_constraints()
            self.lastNameInequalityStart = self.nb_inequality_constraints()

    def nb_equality_constraints(self):
        return self.a_equalities.shape[0]

    def nb_inequality_constraints(self):
        return self.a_inequalities.shape[0]

    def end_constraint_name(self, name):
        if not (name is None or name == ""):
            assert self.lastNameStart == name
            if self.nb_equality_constraints() > self.lastNameEqualityStart:
                self.equalityConstraintNames.append(
                    {
                        "name": name,
                        "start": self.lastNameEqualityStart,
                        "end": self.nb_equality_constraints() - 1,
                    }
                )
            if self.nb_inequality_constraints() > self.lastNameInequalityStart:
                self.inequalityConstraintNames.append(
                    {
                        "name": name,
                        "start": self.lastNameInequalityStart,
                        "end": self.nb_inequality_constraints() - 1,
                    }
                )

    def get_inequality_constraint_name_from_id(self, idv):
        for d in self.inequalityConstraintNames:
            if idv >= d["start"] and id <= d["end"]:
                return d

    def get_equality_constraint_name_from_id(self, idv):
        for d in self.equalityConstraintNames:
            if idv >= d["start"] and id <= d["end"]:
                return d

    def find_inequality_constraints_from_name(self, name):
        constraints = []
        for d in self.inequalityConstraintNames:
            if d["name"] == name:
                constraints.append(d)
        return constraints

    def save(self, filename, force_integer=False):
        self.convertToMosek().save(filename, force_integer=force_integer)

    def save_mps(self, filename):
        assert self.b_lower is None

        f = open(filename, "w")
        f.write("NAME  exportedFromPython\n")
        f.write("ROWS\n")
        f.write(" N  OBJ\n")

        # for i in range(self.b_equalities.size):
        # f.write(' E  E%d\n'%i)
        np.savetxt(f, np.arange(self.b_equalities.size), fmt=" E  E%d", newline="\n")

        # for i in range(self.b_upper.size):
        # f.write(' L  I%d\n'%i)
        np.savetxt(f, np.arange(self.b_upper.size), fmt=" L  I%d", newline="\n")

        f.write("COLUMNS\n")

        a_eq = self.a_equalities.tocsc().tocoo()
        a_ineq = self.a_inequalities.tocsc().tocoo()

        k_eq = 0
        k_ineq = 0
        n_eq_entries = len(a_eq.col)
        n_ineq_entries = len(a_ineq.col)

        for i in range(self.nb_variables):
            f.write("    X%-9dOBJ       %f\n" % (i, self.costsvector[i]))

            while k_eq < n_eq_entries and a_eq.col[k_eq] == i:
                f.write("    X%-9dE%-9d%f\n" % (i, a_eq.ruse_preconditioning))
                k_eq += 1
            while k_ineq < n_ineq_entries and a_ineq.col[k_ineq] == i:
                f.write(
                    "    X%-9dI%-9d%f\n" % (i, a_ineq.row[k_ineq], a_ineq.data[k_ineq])
                )
                k_ineq += 1

        f.write("RHS\n")
        np.savetxt(
            f,
            np.column_stack((np.arange(a_eq.shape[0]), self.b_equalities)),
            fmt="    RHS0      E%-9d%f",
            newline="\n",
        )
        np.savetxt(
            f,
            np.column_stack((np.arange(a_ineq.shape[0]), self.b_upper)),
            fmt="    RHS0      I%-9d%f",
            newline="\n",
        )

        f.write("RANGES\n")
        f.write("BOUNDS\n")
        integer_indices = np.nonzero(self.is_integer)[0]
        np.savetxt(
            f,
            np.column_stack((integer_indices, self.upper_bounds[integer_indices])),
            fmt=" UI bound     X%-9d%f",
            newline="\n",
        )
        np.savetxt(
            f,
            np.column_stack((integer_indices, self.lower_bounds[integer_indices])),
            fmt=" LI bound     X%-9d%f",
            newline="\n",
        )
        continuous_indices = np.nonzero(~self.is_integer)[0]
        np.savetxt(
            f,
            np.column_stack(
                (continuous_indices, self.upper_bounds[continuous_indices])
            ),
            fmt=" UP bound     X%-9d%f",
            newline="\n",
        )
        np.savetxt(
            f,
            np.column_stack(
                (continuous_indices, self.lower_bounds[continuous_indices])
            ),
            fmt=" LO bound     X%-9d%f",
            newline="\n",
        )

        f.write("ENDATA\n")
        f.close()

    def save_ian_e_h_yen(self, folder):
        if self.b_lower is not None:
            print(
                "self.b_lower is not None, you should convert your problem with convert_to_one_sided_inequality_system first"
            )
            raise
        if not np.all(self.lower_bounds == 0):
            print("lower bound constraint on variables should at 0")
            raise

        import os

        a_eq = self.a_equalities.tocoo()
        tmp = np.row_stack(
            (
                [a_eq.shape[0], a_eq.shape[1], 0.0],
                np.column_stack((a_eq.row + 1, a_eq.col + 1, a_eq.data)),
            )
        )
        np.savetxt(os.path.join(folder, "a_eq"), tmp, fmt="%d %d %f")
        np.savetxt(os.path.join(folder, "beq"), self.b_equalities, fmt="%f")
        np.savetxt(os.path.join(folder, "c"), self.costsvector, fmt="%f")
        nb_variables = self.costsvector.size
        upper_bounded = np.nonzero(~np.isinf(self.upper_bounds))[0]
        nb_upper_bounded = len(upper_bounded)
        a_ineq2 = scipy.sparse.coo_matrix(
            (np.ones(nb_upper_bounded), (np.arange(nb_upper_bounded), upper_bounded)),
            (nb_upper_bounded, nb_variables),
        )
        a_ineq = scipy.sparse.vstack((self.a_inequalities, a_ineq2)).tocoo()
        b_upper = np.hstack((self.b_upper, self.upper_bounds[upper_bounded]))
        tmp = np.row_stack(
            (
                [a_ineq.shape[0], a_ineq.shape[1], 0.0],
                np.column_stack((a_ineq.row + 1, a_ineq.col + 1, a_ineq.data)),
            )
        )
        np.savetxt(os.path.join(folder, "A"), tmp, fmt="%d %d %f")
        np.savetxt(os.path.join(folder, "b"), b_upper, fmt="%f")

        with open(os.path.join(folder, "meta"), "w") as f:
            f.write("nb	%d\n" % nb_variables)
            f.write("nf	%d\n" % 0)
            f.write("mI	%d\n" % a_ineq.shape[0])
            f.write("mE	%d\n" % a_eq.shape[0])

    def get_variables_bounds(self):
        types = None
        bl = self.lower_bounds
        bu = self.upper_bounds

        return types, bl, bu

    def add_variables_array(
        self, shape, lower_bounds, upper_bounds, costs=0, name=None, is_integer=False
    ):
        if isinstance(shape, type(0)):
            shape = (shape,)

        nb_variables_added = np.prod(shape)
        indices = np.arange(nb_variables_added).reshape(shape) + self.nb_variables
        self.nb_variables = self.nb_variables + nb_variables_added

        self.a_inequalities._shape = (self.a_inequalities.shape[0], self.nb_variables)
        self.a_equalities._shape = (self.a_equalities.shape[0], self.nb_variables)

        if isinstance(costs, type(0)) or isinstance(costs, type(0.0)):
            v = costs
            costs = np.empty(shape, dtype=np.float)
            costs.fill(v)

        assert np.all(costs.shape == shape)
        lower_bounds, upper_bounds = self.convert_bounds_to_vectors(
            shape, lower_bounds, upper_bounds
        )
        assert np.all(lower_bounds.shape == shape)
        assert np.all(upper_bounds.shape == shape)

        self.upper_bounds = np.append(self.upper_bounds, upper_bounds.ravel())
        self.lower_bounds = np.append(self.lower_bounds, lower_bounds.ravel())
        self.costsvector = np.append(self.costsvector, costs.ravel())
        self.is_integer = np.append(
            self.is_integer, np.full((nb_variables_added), is_integer, dtype=np.bool)
        )

        if name:
            self.variables_dict[name] = indices
        self.set_bounds_on_variables(indices, lower_bounds, upper_bounds)
        return indices

    def convert_bounds_to_vectors(self, shape, lower_bounds, upper_bounds):

        if (
            isinstance(lower_bounds, type(0))
            or isinstance(lower_bounds, type(0.0))
            or isinstance(lower_bounds, np.float64)
        ):
            v = lower_bounds
            lower_bounds = np.empty(shape, dtype=np.float)
            lower_bounds.fill(v)
        if (
            isinstance(upper_bounds, type(0))
            or isinstance(upper_bounds, type(0.0))
            or isinstance(upper_bounds, np.float64)
        ):
            v = upper_bounds
            upper_bounds = np.empty(shape, dtype=np.float)
            upper_bounds.fill(v)

        if upper_bounds is None:
            # assert np.all((lower_bounds.shape==shape))
            upper_bounds = np.empty(shape, dtype=np.float)
            upper_bounds.fill(np.inf)

        if lower_bounds is None:
            # assert np.all((upper_bounds.shape==shape))
            lower_bounds = np.empty(shape, dtype=np.float)
            lower_bounds.fill(-np.inf)

        assert np.all((upper_bounds.shape == shape))
        assert np.all((lower_bounds.shape == shape))

        return lower_bounds, upper_bounds

    def set_bounds_on_variables(self, indices, lower_bounds, upper_bounds):
        # could use task.putboundslice if we were sure that the indices is an increasing sequence n with increments of 1 i.e,n+1,n+2,....n+k
        if isinstance(lower_bounds, type(0)) or isinstance(lower_bounds, type(0.0)):
            self.lower_bounds[indices.ravel()] = lower_bounds
        else:
            self.lower_bounds[indices.ravel()] = lower_bounds.ravel()
        if isinstance(upper_bounds, type(0)) or isinstance(upper_bounds, type(0.0)):
            self.upper_bounds[indices.ravel()] = upper_bounds
        else:
            self.upper_bounds[indices.ravel()] = upper_bounds.ravel()

    def get_variables_indices(self, name):
        """Return the set of indices corresponding to the variables that have have been added with the given name when using add_variables_array"""
        return self.variables_dict[name]

    def set_costs_variables(self, indices, costs):
        assert np.all(costs.shape == indices.shape)
        self.costsvector[indices.ravel()] = costs.ravel()

    def add_equality_constraints_sparse(self, a, b):
        csr_matrix_append_rows(self.a_equalities, a.tocsr())
        self.b_equalities = np.append(self.b_equalities, b)

    def add_inequality_constraints_sparse(
        self, a, lower_bounds=None, upper_bounds=None
    ):
        # add the constraint lower_bounds<=Ax<=upper_bounds to the list of constraints
        # try to use A as a sparse matrix
        # take advantage of the snipy sparse marices to ease things

        if (
            isinstance(lower_bounds, type(0)) or isinstance(lower_bounds, type(0.0))
        ) and lower_bounds == upper_bounds:
            lower_bounds, upper_bounds = self.convert_bounds_to_vectors(
                (a.shape[0],), lower_bounds, upper_bounds
            )
            csr_matrix_append_rows(self.a_equalities, a.tocsr())
            self.b_equalities = np.append(self.b_equalities, lower_bounds)

        else:
            lower_bounds, upper_bounds = self.convert_bounds_to_vectors(
                (a.shape[0],), lower_bounds, upper_bounds
            )
            csr_matrix_append_rows(self.a_inequalities, a.tocsr())
            self.b_lower = np.append(self.b_lower, lower_bounds)
            self.b_upper = np.append(self.b_upper, upper_bounds)

    def add_equality_constraints(self, cols, vals, b):
        """Add a set of equalities to the problem in the form
        y[i] = b for all i
        with y[i]= sum_j vals[i,j]*x[cols[i,j]]
        """
        self.add_inequality_constraints(cols, vals, lower_bounds=b, upper_bounds=b)

    def add_soft_equality_constraints(self, cols, vals, b, coef_penalization):
        """Add a set of soft equalities terms to the problem in the form of
        sum_i abs(coef_penalization[i] * y[i] )
        with y[i]= sum_j vals[i,j]*x[cols[i,j]]
        this is done by adding auxiliary variables.
        """
        self.add_soft_inequality_constraints(
            cols,
            vals,
            lower_bounds=b,
            upper_bounds=b,
            coef_penalization=coef_penalization,
        )

    def add_inequality_constraints(
        self, cols, vals, lower_bounds=None, upper_bounds=None
    ):
        """Add a set of equalities to the problem in the form
        lower_bounds[i] <= y[i] <= upper_bounds[i] for all i
        with y[i]= sum_j vals[i,j]*x[cols[i,j]]
        """
        self.add_soft_inequality_constraints(
            cols,
            vals,
            coef_penalization=np.inf,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def add_soft_inequality_constraints(
        self, cols, vals, coef_penalization, lower_bounds=None, upper_bounds=None
    ):
        """Add a set of "soft" inequalities terms to the problem in the form of
        sum_i abs(coef_penalization[i] * maximum(0, lower_bounds[i] - y[i] , y[i] - upper_bound[i]) )
        with y[i]= sum_j vals[i,j]*x[cols[i,j]]
        this is done by adding auxiliary variables.
        """
        if np.all(coef_penalization == np.inf):
            a = crd_matrix(cols, vals)
            self.add_inequality_constraints_sparse(
                a, lower_bounds=lower_bounds, upper_bounds=upper_bounds
            )

        else:
            if any(coef_penalization == np.inf):
                raise ("penalization with subset with np.inf not handled yet")
            cols, vals = np.broadcast_arrays(cols, vals)
            assert np.all(cols.shape == vals.shape)
            aux = self.add_variables_array(
                (cols.shape[0],),
                upper_bounds=None,
                lower_bounds=0,
                costs=coef_penalization,
            )

            cols2 = np.column_stack((cols, aux))
            assert (upper_bounds is not None) or (lower_bounds is not None)
            if upper_bounds is not None:
                vals2 = np.column_stack((vals, -np.ones((vals.shape[0], 1))))
                self.add_inequality_constraints(
                    cols2, vals2, lower_bounds=None, upper_bounds=upper_bounds
                )
            if lower_bounds is not None:
                vals2 = np.column_stack((vals, np.ones((vals.shape[0], 1))))
                self.add_inequality_constraints(
                    cols2, vals2, lower_bounds, upper_bounds=None
                )
            return aux

    def add_inequalities_pairs(
        self, indices_and_weight_pairs, lower_bounds, upper_bounds, check=True
    ):
        cols = []
        vals = []
        for t in indices_and_weight_pairs:
            cols.append(t[0].flatten())
            # trick to do broadcasting on vals if needed
            vals.append((np.ones(t[0].shape) * t[1]).flatten())
        if isinstance(upper_bounds, np.ndarray):
            upper_bounds = upper_bounds.flatten()
        self.add_linear_constraint_rows(
            np.column_stack(cols), np.column_stack(vals), lower_bounds, upper_bounds
        )
        if (self.solution is not None) and check:
            assert self.check_solution(self.solution)

    def remove_fixed_variables(self):
        # should do a more complete presolve procedure in case we use interior point method
        # http://www.davi.ws/doc/gondzio94presolve.pdf
        free = self.upper_bounds > self.lower_bounds
        id_free = np.nonzero(free)[0]
        nb_free = np.sum(free)
        m_change = scipy.sparse.coo_matrix(
            (np.ones((nb_free)), (id_free, np.arange(nb_free))),
            (self.nb_variables, nb_free),
        )
        shift = np.zeros((self.nb_variables))
        shift[~free] = self.lower_bounds[~free]

        self.b_equalities = self.b_equalities - self.a_equalities * shift
        if self.b_lower is not None:
            self.b_lower = self.b_lower - self.a_inequalities * shift
        if self.b_upper is not None:
            self.b_upper = self.b_upper - self.a_inequalities * shift

        # self.b_equalities equalities= self.b_equalities-self.a_equalities*shift

        self.costsvector = self.costsvector[free]
        b = self.a_inequalities.__dict__["blocks"]
        self.a_inequalities = self.a_inequalities[:, free]
        self.a_inequalities.__dict__["blocks"] = b

        # u,ia,ib=unique_rows(self.a_inequalities.todense())

        b = self.a_equalities.__dict__["blocks"]
        self.a_equalities = self.a_equalities[:, free]
        self.a_equalities.__dict__["blocks"] = b

        # find constraints with single variable left
        # self.a_equalities

        # u,ia,ib=unique_rows(self.a_equalities.todense())
        self.nb_variables = nb_free
        self.lower_bounds = self.lower_bounds[free]
        self.upper_bounds = self.upper_bounds[free]

        # b_equalities= LP.b_equalities
        # b_equalities= LP.b_equalities
        return m_change, shift

    def convert_to_slack_form(self):
        """Convert to the form min_y c.ty Ay=b y>=0 by adding slack variables and shift on x
        the solution of the original problem is obtained using
            x = m_change*y+ shift
        with
            y the solution of the new problem
        have a look at https://ocw.mit.edu/courses/sloan-school-of-management/15-053-optimization-methods-in-management-science-spring-2013/tutorials/MIT15_053S13_tut06.pdf
        """
        self.convert_to_one_sided_inequality_system()

        # inverse variables that are only bounded above using a change of variable x=M*y
        reverse = np.isinf(self.lower_bounds) & (~np.isinf(self.upper_bounds))
        if np.any(reverse):
            raise ("this part of the code has not been tested yet")
            d = np.ones(self.nb_variables)
            d[reverse] = -1
            m1 = scipy.sparse.spdiags([d], [0], self.nb_variables, self.nb_variables)
            a_inequalities = None
            a_equalities = None
            if self.a_inequalities is not None:
                a_inequalities = self.a_inequalities * m1
                a_inequalities.__dict__["blocks"] = [(0, a_inequalities.shape[0] - 1)]
            if self.a_equalities is not None:
                a_equalities = self.a_equalities * m1
                a_equalities.__dict__["blocks"] = [(0, a_equalities.shape[0] - 1)]
            lower_bounds = copy.copy(self.lower_bounds)
            upper_bounds = copy.copy(self.upper_bounds)
            lower_bounds[reverse] = -self.upper_bounds[reverse]
            upper_bounds[reverse] = -self.lower_bounds[reverse]

        else:
            m1 = scipy.sparse.eye(self.nb_variables)
            lower_bounds = copy.copy(self.lower_bounds)
            upper_bounds = copy.copy(self.upper_bounds)
            a_inequalities = copy.copy(self.a_inequalities)
            a_equalities = copy.copy(self.a_equalities)

        # shift lower bounds to 0 by a change of variable y =x-lb
        # Ax=b lb<=x<=ub =>	Ay=A(x-lb)=b-A*lb
        shift = np.zeros(lower_bounds.size)
        shift[~np.isinf(lower_bounds)] = lower_bounds[~np.isinf(lower_bounds)]
        assert self.b_lower is None
        b_upper = self.b_upper - a_inequalities * shift

        if self.b_equalities is not None:
            b_equalities = self.b_equalities - a_equalities * shift
        else:
            b_equalities = None

        upper_bounds = upper_bounds - shift
        lower_bounds = lower_bounds - shift

        # put upper bound constraints into the inequality matrix
        id_upper = np.nonzero(~np.isinf(self.upper_bounds))[0]
        nb_upper = len(id_upper)
        if nb_upper > 0:
            # raise 'this part of the code has not been tested yet'
            t = scipy.sparse.coo_matrix(
                (np.ones(nb_upper), (np.arange(nb_upper), id_upper))
            )
            csr_matrix_append_rows(a_inequalities, t.tocsr())
            b_upper = np.append(b_upper, upper_bounds[id_upper])
        upper_bounds = None

        # replace free variables by a difference of positive variables
        free = np.isinf(-self.lower_bounds) & np.isinf(self.upper_bounds)
        # create the permutation matrix that set the all free variables after the other variables
        nb_free = np.sum(free)
        nb_variables = self.nb_variables
        costsvector = self.costsvector
        if nb_free > 0:
            # raise 'this part of the code has not been tested yet'
            nb_not_free = nb_variables - nb_free
            j_mat = (np.cumsum(~free) - 1) * (~free) + (
                np.cumsum(free) + nb_not_free - 1
            ) * (free)
            perm = scipy.sparse.coo_matrix(
                (np.ones(self.nb_variables), (np.arange(self.nb_variables), j_mat))
            )
            tmp = scipy.sparse.vstack(
                (
                    scipy.sparse.hstack(
                        (
                            scipy.sparse.eye(nb_not_free),
                            scipy.sparse.coo_matrix((nb_not_free, 2 * nb_free)),
                        )
                    ),
                    scipy.sparse.hstack(
                        (
                            scipy.sparse.coo_matrix((nb_free, nb_not_free)),
                            scipy.sparse.eye(nb_free),
                            -scipy.sparse.eye(nb_free),
                        )
                    ),
                )
            )
            m2 = perm * tmp
            m_change = m1 * m2
            nb_variables = nb_not_free + 2 * nb_free
            lower_bounds = np.zeros(nb_variables)
            costsvector = costsvector * m_change
            if a_equalities is not None:
                a_equalities = a_equalities * m_change
                a_equalities.__dict__["blocks"] = [(0, a_equalities.shape[0] - 1)]
            if a_inequalities is not None:
                a_inequalities = a_inequalities * m_change
                a_inequalities.__dict__["blocks"] = [(0, a_inequalities.shape[0] - 1)]
        else:
            m_change = m1

        # remove lower inequality constraints

        # replace inequality constraint Ax<=b  by ax+s=b s>=0
        nbslack = a_inequalities.shape[0]

        nb_variables = nb_variables + nbslack
        a_inequalities = scipy.sparse.hstack(
            (a_inequalities, scipy.sparse.eye(nbslack))
        )
        a_equalities._shape = (a_equalities.shape[0], nb_variables)
        m_change = m_change.tocsr()
        m_change._shape = (m_change.shape[0], nb_variables)

        lower_bounds = np.append(lower_bounds, np.zeros(nbslack))
        costsvector = np.append(costsvector, np.zeros(nbslack))

        csr_matrix_append_rows(a_equalities, a_inequalities.tocsr())
        b_equalities = np.append(b_equalities, b_upper)

        b_lower = None
        b_upper = None
        a_inequalities = None

        self.nb_variables = nb_variables
        self.b_lower = b_lower
        self.b_upper = b_upper
        self.a_inequalities = a_inequalities
        self.b_equalities = b_equalities
        self.a_equalities = a_equalities
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.costsvector = costsvector

        return m_change, shift

    def convert_to_all_equalities(self):
        """Convert to the form min c.t Ax=b lb<=x<=ub by adding slack variables
        the solution of th original problem is obtained using the first elements in x.
        """
        if self.a_inequalities is not None:
            m = self.a_inequalities.shape[0]
            n = self.a_inequalities.shape[1]
            self.add_variables_array(m, self.b_lower, self.b_upper)
            self.a_inequalities._shape = (self.a_inequalities.shape[0], n)
            self.add_inequality_constraints_sparse(
                scipy.sparse.hstack((self.a_inequalities, -scipy.sparse.eye(m))), 0, 0
            )
            self.b_lower = None
            self.b_upper = None
            self.a_inequalities = None

    def convert_to_one_sided_inequality_system(self):
        """Convert to the form min c.t a_ineq*x<=b_ineq a_eq*x=b lb<=x<=ub by augmenting the size of a_ineq."""
        if (self.a_inequalities is not None) and (self.b_lower is not None):
            idskeep_upper = np.nonzero(self.b_upper != np.inf)[0]
            mapping_upper = np.hstack(([0], np.cumsum(self.b_upper != np.inf)))
            idskeep_lower = np.nonzero(self.b_lower != -np.inf)[0]
            mapping_lower = np.hstack(([0], np.cumsum(self.b_lower != np.inf)))
            if len(idskeep_lower) > 0 and len(idskeep_upper) > 0:

                new_inequality_constraint_names = []
                for d in self.inequalityConstraintNames:
                    d = {
                        "name": d["name"],
                        "start": mapping_upper[d["start"]],
                        "end": mapping_upper[d["end"]],
                    }
                    new_inequality_constraint_names.append(d)
                for d in self.inequalityConstraintNames:

                    d = {
                        "name": d["name"],
                        "start": idskeep_upper.size + mapping_lower[d["start"]],
                        "end": idskeep_upper.size + mapping_lower[d["end"]],
                    }
                    new_inequality_constraint_names.append(d)

                self.inequalityConstraintNames = new_inequality_constraint_names
                self.a_inequalities = scipy.sparse.vstack(
                    (
                        self.a_inequalities[idskeep_upper, :],
                        -self.a_inequalities[idskeep_lower, :],
                    )
                ).tocsr()

            elif len(idskeep_lower) > 0:
                self.a_inequalities = -self.a_inequalities
            else:
                self.a_inequalities = self.a_inequalities
            self.a_inequalities.__dict__["blocks"] = [
                (0, self.a_inequalities.shape[0] - 1)
            ]
            self.b_upper = np.hstack(
                (self.b_upper[idskeep_upper], -self.b_lower[idskeep_lower])
            )
            self.b_lower = None

    def convert_to_all_inequalities(self):
        """Convert to the form min c.t b_lower<=a_ineq*x<=b_upper lb<=x<=ub by augmenting the size of a_ineq."""
        if self.a_equalities is not None:

            if self.b_lower is None:
                self.b_lower = np.full((self.a_inequalities.shape[0]), -np.inf)
            if self.b_upper is None:
                self.b_upper = np.full((self.a_inequalities.shape[0]), np.inf)

            new_inequality_constraint_names = []
            for d in self.equalityConstraintNames:
                new_inequality_constraint_names.append(d)
            for d in self.inequalityConstraintNames:

                d = {
                    "name": d["name"],
                    "start": self.a_equalities.shape[0] + d["start"],
                    "end": self.a_equalities.shape[0] + d["end"],
                }
                new_inequality_constraint_names.append(d)
            self.inequalityConstraintNames = new_inequality_constraint_names
            self.equalityConstraintNames = []

            self.a_inequalities = scipy.sparse.vstack(
                (self.a_equalities, self.a_inequalities)
            )

            self.b_lower = np.hstack((self.b_equalities, self.b_lower))
            self.b_upper = np.hstack((self.b_equalities, self.b_upper))
            self.a_equalities = None
            self.b_equalities = None

    def convert_to_all_inequalities_without_bounds(self):
        """Convert to the form min c.t b_lower<=a_ineq*x<=b_upper by augmenting the size of a_ineq."""
        if self.b_lower is None:
            self.b_lower = np.full((self.a_inequalities.shape[0]), -np.inf)
        if self.b_upper is None:
            self.b_upper = np.full((self.a_inequalities.shape[0]), np.inf)

        self.convert_to_all_inequalities()
        non_free_ids = np.nonzero(
            ~(np.isinf(self.lower_bounds) & np.isinf(self.upper_bounds))
        )[0]
        nb_non_free_ids = len(non_free_ids)
        eye_reduced = scipy.sparse.coo_matrix(
            (np.ones(nb_non_free_ids), (np.arange(nb_non_free_ids), non_free_ids)),
            (nb_non_free_ids, self.nb_variables),
        )

        self.a_inequalities = scipy.sparse.vstack(
            (self.a_inequalities, eye_reduced)
        ).tocsr()

        self.b_lower = np.hstack((self.b_lower, self.lower_bounds[non_free_ids]))
        self.b_upper = np.hstack((self.b_upper, self.upper_bounds[non_free_ids]))
        self.lower_bounds.fill(-np.inf)
        self.upper_bounds.fill(np.inf)

    def convert_to_single_inequalities_without_bounds(self):
        """Convert to the form min c.t a_ineq*x<=b_upper by augmenting the size of a_ineq."""
        self.convert_to_all_inequalities_without_bounds()
        self.convert_to_one_sided_inequality_system()

    def lexsort_constraints(self):
        # reorder constraints in lexicographic order
        if self.a_inequalities is not None:
            row_order = np.lexsort(np.fliplr(self.a_inequalities.todense()).T).squeeze(
                0
            )
            self.a_inequalities = self.a_inequalities[row_order, :]
            if self.b_upper is not None:
                self.b_upper = self.b_upper[row_order]
            if self.b_lower is not None:
                self.b_lower = self.b_lower[row_order]
        if self.a_equalities is not None:
            row_order = np.lexsort(np.fliplr(self.a_equalities.todense()).T).squeeze(0)
            self.a_inequalities = self.a_inequalities[row_order, :]
            self.b_equalities = self.b_equalities[row_order]

    def convert_to_cvxpy(self):

        if not (self.a_inequalities is None) and self.a_inequalities.shape[0] > 0:
            check_csr_matrix(self.a_inequalities)
            a_ineq = self.a_inequalities
        else:
            a_ineq = None
        if self.a_equalities.shape[0] > 0:
            a_eq = self.a_equalities
            b_eq = self.b_equalities
        else:
            a_eq = None
            b_eq = None

        # uses cvxpy to call SCS as it make its easier to specify tje problem
        # Problem data.
        # Construct the problem.
        x = cvxpy.Variable(self.nb_variables)
        objective = cvxpy.Minimize(np.matrix(self.costsvector[None, :]) * x)

        constraints = []

        if np.all(np.isinf(self.lower_bounds)):
            pass
        elif np.any(np.isinf(self.lower_bounds)):
            print("not code yet")
            raise
        else:
            constraints.append(self.lower_bounds <= x)

        if np.all(np.isinf(self.upper_bounds)):
            pass
        elif np.any(np.isinf(self.upper_bounds)):
            print("not code yet")
            raise
        else:
            constraints.append(x <= self.upper_bounds)

        if a_ineq is not None:
            if self.b_upper is not None:
                if np.all(np.isinf(self.b_upper)):
                    pass
                elif np.any(np.isinf(self.b_upper)):
                    print("not yet coded")
                    raise
                else:
                    constraints.append(a_ineq * x <= self.b_upper)
            if self.b_lower is not None:
                if np.all(np.isinf(self.b_lower)):
                    pass
                elif np.any(np.isinf(self.b_lower)):
                    print("not yet coded")
                    raise
                else:
                    constraints.append(self.b_lower <= a_ineq * x)
        if a_eq is not None:
            constraints.append(a_eq * x == b_eq)
        prob = cvxpy.Problem(objective, constraints)
        return prob, x

    def has_bounds(self):
        return (not np.all(np.isinf(self.lower_bounds) & (self.lower_bounds < 0))) or (not np.all(np.isinf(self.upper_bounds) & (self.upper_bounds > 0)))

    def all_bounded(self):
        return not(np.any(np.isinf(self.lower_bounds)) or np.any(np.isinf(self.upper_bounds)))

    def has_single_inequalities_without_bounds(self):
        return not(self.has_bounds() or(self.a_equalities is not None) or(self.b_lower is not None))

    def get_dual(self):
        if not self.has_single_inequalities_without_bounds() :
            raise BaseException(
                'Please convert your problem using convert_to_single_inequalities_without_bounds')

        # min c.x  s.t. a_ineq*x<=b_upper
        #  = min_x (max_y (c.x  + y*(a_ineq*x-b_upper) s.t y>=0))
        #  = max_y (min_x (c.x  + y*(a_ineq*x-b_upper)) s.t y>=0)
        #  = max_y (min_x (c.x  + y*a_ineq*x-y * b_upper)) s.t y>=0
        #  = max_y (min_x ((c+y*a_ineq)*x -y * b_upper)) s.t y>=0
        #  = min_y (max_x (-(c+y*a_ineq)*x + y * b_upper)) s.t y>=0
        #  = min_y y*b_upper s.t a_ineq.T * y= -c  y>=0

        lp_dual = SparseLP()
        lp_dual.add_variables_array(
            (self.a_inequalities.shape[0]), costs=self.b_upper, lower_bounds=0, upper_bounds=None)
        lp_dual.add_equality_constraints_sparse(
            self.a_inequalities.T, -self.costsvector)
        return lp_dual

    def solve(
        self,
        method="admm",
        get_timing=True,
        x0=None,
        nb_iter=10000,
        max_duration=None,
        callback_func=None,
        nb_iter_plot=10,
        plot_solution=None,
        ground_truth=None,
        ground_truth_indices=None,
        method_options=None,
        error_if_fail=False,
    ):

        if not (self.a_inequalities is None) and self.a_inequalities.shape[0] > 0:
            check_csr_matrix(self.a_inequalities)
            a_ineq = self.a_inequalities
        else:
            a_ineq = None
        if self.a_equalities.shape[0] > 0:
            a_eq = self.a_equalities
            b_eq = self.b_equalities
        else:
            a_eq = None
            b_eq = None

        start = time.clock()

        self.distance_to_ground_truth = []
        self.distanceToGroundTruthAfterRounding = []
        self.opttime_curve = []
        self.dopttime_curve = []
        self.pobj_curve = []
        self.dobj_curve = []
        self.pobjbound = []
        self.max_violated_inequality = []
        self.max_violated_equality = []
        self.max_violated_constraint = []
        self.itrn_curve = []

        def scipy_call_back(solution, **kwargs):
            if ground_truth is not None:
                self.distance_to_ground_truth.append(
                    np.mean(np.abs(ground_truth - solution["x"][ground_truth_indices]))
                )
                self.distanceToGroundTruthAfterRounding.append(
                    np.mean(
                        np.abs(
                            ground_truth - np.round(solution["x"][ground_truth_indices])
                        )
                    )
                )
            duration = time.clock() - start
            self.opttime_curve.append(duration)
            self.pobj_curve.append(self.costsvector.dot(solution["x"].T))
            maxv = self.max_constraint_violation(solution["x"])
            self.max_violated_constraint.append(maxv)

        def simplex_call_back(solution, **kwargs):
            if ground_truth is not None:
                self.distance_to_ground_truth.append(
                    np.mean(np.abs(ground_truth - solution[ground_truth_indices]))
                )
                self.distanceToGroundTruthAfterRounding.append(
                    np.mean(
                        np.abs(ground_truth - np.round(solution[ground_truth_indices]))
                    )
                )
            duration = time.clock() - start
            self.opttime_curve.append(duration)
            self.pobj_curve.append(self.costsvector.dot(solution.T))
            maxv = self.max_constraint_violation(solution)
            self.max_violated_constraint.append(maxv)

        def callback_func(
            niter,
            solution,
            energy1,
            energy2,
            duration,
            max_violated_equality,
            max_violated_inequality,
            is_active_variable=None,
        ):
            if ground_truth is not None:
                self.distance_to_ground_truth.append(
                    np.mean(np.abs(ground_truth - solution[ground_truth_indices]))
                )
                self.distanceToGroundTruthAfterRounding.append(
                    np.mean(
                        np.abs(ground_truth - np.round(solution[ground_truth_indices]))
                    )
                )
            self.itrn_curve.append(niter)
            self.opttime_curve.append(duration)
            self.dopttime_curve.append(duration)
            self.dobj_curve.append(energy2)
            self.pobj_curve.append(energy1)
            maxv = self.max_constraint_violation(solution)
            self.max_violated_constraint.append(maxv)
            self.max_violated_equality.append(max_violated_equality)
            self.max_violated_inequality.append(max_violated_inequality)
            if plot_solution is not None:
                plot_solution(niter, solution, is_active_variable=is_active_variable)

        if method not in solving_methods:
            raise BaseException(
                f"method {method} not valid. Avalaible method are {', '.join(solving_methods)}"
            )

        if method in ["scipy_simplex", "scipy_revised_simplex", "scipy_interior_point"]:

            if not (self.b_lower is None) and not (
                np.all(np.isinf(self.b_lower) & (self.b_lower < 0))
            ):

                raise BaseException(
                    "you need to convert your lp to a one sided inequality system using convert_to_one_sided_inequality_system"
                )
            if a_eq is None:
                a_eq = None
                b_eq = None
            else:
                a_eq = a_eq.toarray()
                b_eq = b_eq
            if a_ineq is None:
                a_ineq = None
            else:
                a_ineq = a_ineq.toarray()

            method_map = {
                "scipy_simplex": "simplex",
                "scipy_revised_simplex": "revised simplex",
                "scipy_interior_point": "interior-point",
            }
            sol = scipy.optimize.linprog(
                self.costsvector,
                A_ub=a_ineq,
                b_ub=self.b_upper,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=np.column_stack((self.lower_bounds, self.upper_bounds)),
                method=method_map[method],
                callback=scipy_call_back,
            )
            if error_if_fail and not sol["success"]:
                raise BaseException(sol["message"])
            x = sol["x"]

        elif method == "mehrotra":

            lp_slack = copy.deepcopy(self)
            (
                m_change1,
                shift1,
            ) = lp_slack.remove_fixed_variables()  # removed fixed variables
            m_change2, shift2 = lp_slack.convert_to_slack_form()

            def mehrotra_call_back(solution, niter, **kwargs):
                x = m_change2 * solution - shift2
                x = m_change1 * x - shift1
                self.itrn_curve.append(niter)
                simplex_call_back(x)

            f, x, y, s, n = mpc_sol(
                lp_slack.a_equalities,
                lp_slack.b_equalities,
                lp_slack.costsvector,
                callback=mehrotra_call_back,
            )
            lp_slack.check_solution(x)
            x = m_change2 * x - shift2
            x = m_change1 * x - shift1
            self.check_solution(x)
            self.max_constraint_violation(x)

        elif method == "CVXOPT":

            prob, x = self.convert_to_cvxpy()
            # The optimal objective is returned by prob.solve().
            result = prob.solve(verbose=True, solver=cvxpy.CVXOPT)
            x = np.array(x.value).flatten()

            # from cvxopt import matrix, solvers
            # the format axcpt
            # lp_slack=copy.deepcopy(self)
            # m_change,shift=lp_slack.convert_to_slack_form()
            # sol=solvers.lp(c, G, h, A , b)  # uses cvxopt conic solver
            # need to rewrite the problem in the form
            # minimize    c'*x
            # subject to  G*x <= h and A*x = b
            # print 'not yet coded'
            # raise

        elif method == "SCS":
            prob, x = self.convert_to_cvxpy()
            # The optimal objective is returned by prob.solve().
            result = prob.solve(
                verbose=True, solver=cvxpy.SCS, max_iters=10000, eps=1e-5
            )
            x = np.array(x.value).flatten()

        elif method == "ECOS":
            prob, x = self.convert_to_cvxpy()
            # The optimal objective is returned by prob.solve().
            result = prob.solve(verbose=True, solver=cvxpy.ECOS)
            x = np.array(result).flatten()

        elif method == "admm":
            x = lp_admm(
                self.costsvector,
                a_eq,
                b_eq,
                a_ineq,
                self.b_lower,
                self.b_upper,
                self.lower_bounds,
                self.upper_bounds,
                nb_iter=nb_iter,
                x0=x0,
                callback_func=callback_func,
                max_duration=max_duration,
                nb_iter_plot=nb_iter_plot,
            )

        elif method == "admm_blocks":
            x = lp_admm_block_decomposition(
                self.costsvector,
                a_eq,
                b_eq,
                a_ineq,
                self.b_lower,
                self.b_upper,
                self.lower_bounds,
                self.upper_bounds,
                nb_iter=nb_iter,
                nb_iter_plot=nb_iter_plot,
                x0=x0,
                callback_func=callback_func,
                max_duration=max_duration,
            )
        elif method == "admm2":
            x = lp_admm2(
                self.costsvector,
                a_eq,
                b_eq,
                a_ineq,
                self.b_lower,
                self.b_upper,
                self.lower_bounds,
                self.upper_bounds,
                nb_iter=nb_iter,
                x0=x0,
                callback_func=callback_func,
                max_duration=max_duration,
                nb_iter_plot=nb_iter_plot,
            )

        elif method == "chambolle_pock_ppd":
            lp_reduced = copy.deepcopy(self)
            (
                m_change1,
                shift1,
            ) = lp_reduced.remove_fixed_variables()  # removed fixed variables

            def this_back(
                niter,
                solution,
                energy1,
                energy2,
                duration,
                max_violated_equality,
                max_violated_inequality,
            ):
                solution = m_change1 * solution - shift1
                callback_func(
                    niter,
                    solution,
                    energy1,
                    energy2,
                    duration,
                    max_violated_equality,
                    max_violated_inequality,
                )

            x, best_integer_solution = chambolle_pock_ppd(
                lp_reduced.costsvector,
                lp_reduced.a_equalities,
                lp_reduced.b_equalities,
                lp_reduced.a_inequalities,
                lp_reduced.b_lower,
                lp_reduced.b_upper,
                lp_reduced.lower_bounds,
                lp_reduced.upper_bounds,
                x0=None,
                alpha=1,
                theta=1,
                nb_max_iter=nb_iter,
                callback_func=this_back,
                max_duration=max_duration,
                save_problem=False,
                nb_iter_plot=nb_iter_plot,
            )
            x = m_change1 * x - shift1

        elif method == "chambolle_pock_linesearch":

            lp2 = copy.deepcopy(self)
            lp2.convert_to_single_inequalities_without_bounds()

            # lp2.lexsort_constraints()

            def this_call_back(x, y, niter, **kwargs):
                self.itrn_curve.append(niter)
                # use y instead of x because we are solving dual in chambolle_pock_linesearch
                simplex_call_back(y)

            # solving the dual
            # method="standard"

            options = {
                "method": "standard",
                "eps": 1e-15,
                "tol": 1e-15,
                "nmax": nb_iter,
                "y_sol": ground_truth,
            }
            if method_options is not None:
                for k, v in method_options.items():
                    options[k] = v

            options["max_duration"] = max_duration
            options["callback"] = this_call_back

            sol = chambolle_pock_linesearch(
                lp2.a_inequalities.T, -lp2.costsvector, lp2.b_upper, **options
            )
            x = sol["y"]

            lp2.check_solution(x)
            self.check_solution(x)
            self.max_constraint_violation(x)

        elif method == "dual_gradient_ascent":
            x, y_eq, y_ineq = dual_gradient_ascent(
                x=x0,
                lp=self,
                nb_max_iter=nb_iter,
                callback_func=callback_func,
                y_eq=None,
                y_ineq=None,
                max_duration=max_duration,
                nb_iter_plot=nb_iter_plot,
            )
        elif method == "dual_coordinate_ascent":
            lp_reduced = copy.deepcopy(self)
            (
                m_change1,
                shift1,
            ) = lp_reduced.remove_fixed_variables()  # removed fixed variables

            def this_back(
                niter,
                solution,
                energy1,
                energy2,
                duration,
                max_violated_equality,
                max_violated_inequality,
            ):
                solution = m_change1 * solution - shift1
                callback_func(
                    niter,
                    solution,
                    energy1,
                    energy2,
                    duration,
                    max_violated_equality,
                    max_violated_inequality,
                )

            x, y_eq, y_ineq = dual_coordinate_ascent(
                x=None,
                lp=lp_reduced,
                nb_max_iter=nb_iter,
                callback_func=this_back,
                y_eq=None,
                y_ineq=None,
                max_duration=max_duration,
                nb_iter_plot=nb_iter_plot,
            )
            x = m_change1 * x - shift1

        elif method == "osqp":
            lp_osqp_form = copy.deepcopy(self)
            lp_osqp_form.convert_to_all_inequalities_without_bounds()
            b_lower = lp_osqp_form.b_lower
            b_lower = np.maximum(-1000, b_lower)
            b_upper = lp_osqp_form.b_upper
            b_upper = np.minimum(1000, b_upper)
            p = scipy.sparse.csc_matrix((self.nb_variables, self.nb_variables))

            opts = {
                "verbose": True,
                "eps_abs": 1e-09,
                "eps_rel": 1e-09,
                "max_iter": nb_iter,
                "rho": 0.1,
                "adaptive_rho": False,
                "polish": True,
                "check_termination": 1,
                "warm_start": False,
            }

            model = osqp.OSQP()
            model.setup(
                p,
                lp_osqp_form.costsvector,
                lp_osqp_form.a_inequalities.tocsc(),
                b_lower,
                b_upper,
                **opts,
            )
            res = model.solve()
            x = res.x
            simplex_call_back(x)
            self.itrn_curve = [res.info.iter]

        else:
            print("unkown LP solver method " + method)
            raise
        elapsed = time.clock() - start

        if get_timing:
            return x, elapsed
        else:
            return x
