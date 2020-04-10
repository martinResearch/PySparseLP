"""Interior Point LP Solver.

For reference, please refer to "On the Implementation of a Primal-Dual
Interior Point Method" by Sanjay Mehrotra.
https://www.researchgate.net/publication/230873223_On_the_Implementation_of_a_Primal-Dual_Interior_Point_Method
this code is largely inspired from https://github.com/YimingYAN/mpc.
"""
import numpy as np
from numpy.linalg import norm

from scipy import sparse
from scipy.sparse.linalg import spsolve


from .xorshift import XorShift


def initial_point(a, b, c, use_umfpack=False):

    n = a.shape[1]
    e = np.ones((n,))

    # solution for min norm(s) s.t. A'*y + s = c
    # y =sparse.linalg.cg(A*A.T, A*c,tol=1e-7)[0]
    y = spsolve(a * a.T, a * c, use_umfpack=use_umfpack)

    # y2 =sparse.linalg.cgs(A*A.T, A*c)[0]
    # y2 =sparse.linalg.gmres(A*A.T, A*c,)[0]

    s = c - a.T * y

    # solution for min norm(x) s.t. Ax = b
    x = a.T * spsolve(a * a.T, b, use_umfpack=use_umfpack)
    # x = A.T*sparse.linalg.cg(A*A.T, b,tol=1e-7)[0]

    # delta_x and delta_s
    delta_x = max(-1.5 * np.min(x), 0)
    delta_s = max(-1.5 * np.min(s), 0)

    # delta_x_c and delta_s_c
    pdct = 0.5 * (x + delta_x * e).dot(s + delta_s * e)
    delta_x_c = delta_x + pdct / (np.sum(s) + n * delta_s)
    delta_s_c = delta_s + pdct / (np.sum(x) + n * delta_x)

    # output
    x0 = x + delta_x_c * e
    s0 = s + delta_s_c * e
    y0 = y
    return x0, y0, s0


def newton_direction(r_b, r_c, r_x_s, a, m, n, x, s, lu, error_check=0):

    rhs = np.hstack((-r_b, -r_c + r_x_s / x))
    d_2 = -np.minimum(1e16, s / x)
    b = sparse.vstack(
        (
            sparse.hstack((sparse.coo_matrix((m, m)), a)),
            sparse.hstack((a.T, sparse.diags([d_2], [0]))),
        )
    )

    # ldl' factorization
    # if L and D are not provided, we calc new factorization; otherwise,
    # reuse them
    use_lu = True
    if use_lu:
        if lu is None:
            lu = sparse.linalg.splu(b.tocsc())
            # wikipedia says it uses Mehrotra cholesky but the matrix i'm getting is not definite positive
            # scikits.sparse.cholmod.cholesky fails without a warning

        sol = lu.solve(rhs)
    else:
        sol = sparse.linalg.cg(b, rhs, tol=1e-5)[0]
        # assert(np.max(np.abs(B*sol-rhs))<1e-5)

    dy = sol[:m]
    dx = sol[m : m + n]
    ds = -(r_x_s + s * dx) / x

    if error_check == 1:
        print(
            "error = %6.2e"
            % (
                norm(a.T * dy + ds + r_c)
                + norm(a * dx + r_b)
                + norm(s * dx + x * ds + r_x_s)
            ),
        )
        print("\t + err_d = %6.2e" % (norm(a.T * dy + ds + r_c)),)
        print("\t + err_p = %6.2e" % (norm(a * dx + r_b)),)
        print("\t + err_gap = %6.2e\n" % (norm(s * dx + x * ds + r_x_s)),)

    return dx, dy, ds, lu


def step_size(x, s, d_x, d_s, eta=0.9995):
    alpha_x = -1 / min(min(d_x / x), -1)
    alpha_x = min(1, eta * alpha_x)
    alpha_s = -1 / min(min(d_s / s), -1)
    alpha_s = min(1, eta * alpha_s)
    return alpha_x, alpha_s


def mpc_sol(
    a,
    b,
    c,
    max_iter=100,
    eps=1e-13,
    theta=0.9995,
    verbose=2,
    error_check=False,
    callback=None,
):

    a = sparse.coo_matrix(a)
    c = np.squeeze(np.array(c))
    b = np.squeeze(np.array(b))

    # Initialization

    m, n = a.shape
    alpha_x = 0
    alpha_s = 0

    if verbose > 1:
        print(
            "\n%3s %6s %9s %11s %9s %9s %9s\n" % (
                "ITER", "COST", "MU", "RESIDUAL", "ALPHAX", "ALPHAS", "MAXVIOL")
        )

    # Choose initial point
    x, y, s = initial_point(a, b, c)

    bc = 1 + max([norm(b), norm(c)])

    # Start the loop
    niter_done = 0

    for niter in range(max_iter):
        # Compute residuals and update mu
        r_b = a * x - b
        r_c = a.T * y + s - c
        r_x_s = x * s
        mu = np.mean(r_x_s)
        f = c.T.dot(x)

        # Check relative decrease in residual, for purposes of convergence test
        residual = norm(np.hstack((r_b, r_c, r_x_s)) / bc)

        if verbose > 1:
            maxviol = max(np.max(np.abs(r_b)), np.max(-x))
            print("%3d %9.2e %9.2e %9.2e %9.4g %9.4g %9.2e" %
                  (niter, f, mu, residual, alpha_x, alpha_s, maxviol))

        if callback is not None:
            callback(x, niter)

        if residual < eps:
            break

        # ----- Predictor step -----

        # Get affine-scaling direction
        dx_aff, dy_aff, ds_aff, lu = newton_direction(
            r_b, r_c, r_x_s, a, m, n, x, s, None, error_check
        )

        # Get affine-scaling step length
        alpha_x_aff, alpha_s_aff = step_size(x, s, dx_aff, ds_aff, 1)
        mu_aff = (x + alpha_x_aff * dx_aff).dot(s + alpha_s_aff * ds_aff) / n

        # Set central parameter
        sigma = (mu_aff / mu) ** 3

        # ----- Corrector step -----

        # Set up right hand sides
        r_x_s = r_x_s + dx_aff * ds_aff - sigma * mu * np.ones((n))

        # Get corrector's direction
        dx_cc, dy_cc, ds_cc, lu = newton_direction(
            r_b, r_c, r_x_s, a, m, n, x, s, lu, error_check
        )

        # Compute search direction and step
        dx = dx_aff + dx_cc
        dy = dy_aff + dy_cc
        ds = ds_aff + ds_cc

        alpha_x, alpha_s = step_size(x, s, dx, ds, theta)

        # Update iterates
        x = x + alpha_x * dx
        y = y + alpha_s * dy
        s = s + alpha_s * ds

        if niter == max_iter and verbose > 1:
            print("max_iter reached!\n")
        niter_done = niter

    if verbose > 0:
        print("\nDONE! [m,n] = [%d, %d], N = %d\n" % (m, n, niter))

    f = c.T.dot(x)

    return f, x, y, s, niter_done


if __name__ == "__main__":

    m = 100
    n = 120

    r = XorShift()
    a = np.matrix(r.randn(m, n))
    b = a * r.rand(n, 1)
    c = a.T * r.rand(m, 1)
    c = c + r.rand(n, 1)
    f, x, y, s, N = mpc_sol(a, b, c)
