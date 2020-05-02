"""Primal dual Chambolle and Pock algorithm with line search
Translated from matlab code written in 2012 by Mengqi(Mandy) Xia and Kevin Musgrave
as final project for Cornell's CS6820 Analysis of Algorithms Class
Oringal code: https://github.com/xiamengqi2012/ChambollePockLinesearch
This is probably implementing the method described in
A First-Order Primal-Dual Algorithm with Linesearch. Yura Malitsky and  ThomasPock. SIAM Journal on Optimization. 2018
https://arxiv.org/abs/1608.08883
It could probably be accelerated using some preconditioning.
"""
import time

import numpy as np

import scipy.sparse as sparse


methods = ("standard", "xyseparate", "without_linesearch")


def sparse_matrix_norm(a):
    np.random.seed(0)
    v0 = np.random.rand(a.shape[0])
    return sparse.linalg.svds(a, k=1, which="LM", v0=v0)[1][
        0
    ]  # intialization is random by default, which mak ethe code non determinisic


def chambolle_pock_linesearch(
    a,
    b,
    c,
    nmax=10000000,
    eps=1e-9,
    tol=1e-9,
    y_sol=None,
    method="standard",
    callback=None,
    max_duration=np.inf,
    stage=True,
):
    """Solve a primal-dual pair min{c'x|Ax=b,x>=0} and max{b'y|A'y<=c} using Chambolle and Pock algorithm.
    A is a matrix. b and c are column vectors. Input x and y are current x
    and y values. gamma and tau are two user defined parameters for CP algorithm
    Return x and y are updated values after one iteration

    Translated from matlab code written in 2012 by Mengqi(Mandy) Xia and Kevin Musgrave
    as final project for Cornell's CS6820 Analysis of Algorithms Class
    original code: https://github.com/xiamengqi2012/ChambollePockLinesearch
    it probably implements the method described in
    A First-Order Primal-Dual Algorithm with Linesearch.
    Yura Malitsky and  ThomasPock. SIAM Journal on Optimization. 2018
    https://arxiv.org/abs/1608.08883
    It could probably be accelerated using some preconditioning
    """

    print(f"method ={method}")
    start = time.clock()
    gamma = 1
    tau = 1 / (2 * (sparse_matrix_norm(a) ** 2) * gamma)
    # Restriction: gamma*tau < 1/norm(A)^2

    err = np.inf
    values = []
    errors = []

    nr, nc = a.shape

    x = np.zeros((nc))
    y = np.zeros((nr))

    xnew = np.zeros((nc))
    ynew = np.zeros((nr))

    n_iter = 1
    alpha_0 = 1  # The minimum value of alpha
    alpha_max = 50  # Alpha value that we start with
    factor = (
        1 / 1.4
    )  # Backtracking linesearch is updated each time using alpha=alpha*factor

    a_t_csr = a.T.tocsr()
    a = a.tocsr()

    q11 = sparse.eye(nr) / gamma
    q12 = -a
    q21 = -a.T
    q22 = sparse.eye(nc) / tau
    q = sparse.bmat([[q11, q12], [q21, q22]]).tocsr()

    elapsed = 0

    alpha = alpha_max

    factor_augment = (
        1.3 / factor
    )  # not in original implementation , but seem to speed things up

    while err > tol and n_iter < nmax and elapsed < max_duration:

        n_iter += 1
        x = xnew
        y = ynew
        dist_sol_list = []

        xnew, ynew, vk, num_activate, alpha = chambolle_pock_update(
            a,
            a_t_csr,
            b,
            c,
            x,
            y,
            gamma,
            tau,
            method,
            alpha_0,
            min(alpha * factor_augment, alpha_max),
            factor,
            q,
            eps,
            stage,
        )
        value = b.dot(ynew)
        values.append(value)
        err = np.sqrt(
            np.linalg.norm(a * xnew - b) ** 2
            + np.linalg.norm(np.maximum(a_t_csr * ynew - c, 0)) ** 2
        )

        if y_sol is not None:
            dist_sol = np.mean(abs(y - y_sol))
            dist_sol_list.append(dist_sol)
        errors.append(err)

        if n_iter % 100 == 0:
            elapsed = time.clock() - start
            if y_sol is not None:
                print(
                    f"iteration {n_iter} time ={elapsed:2.1f} sec value={value} error ={err:1.3e} dist to sol={dist_sol:1.3e}"
                )
            else:
                print(f"iteration {n_iter} time ={elapsed:2.0f} sec error ={err:1.3e}")

        if callback is not None:
            callback(x, y, n_iter)

    return {
        "x": xnew,
        "y": ynew,
        "values": values,
        "errors": errors,
        "dist_sol": dist_sol_list,
    }


def chambolle_pock_update(
    a,
    a_t_csr,
    b,
    c,
    x,
    y,
    gamma,
    tau,
    method,
    alpha_0,
    alpha_max,
    factor,
    q,
    eps,
    stage,
):

    assert method in methods

    ax_minus_b = a * x - b
    num_activate = 0
    gamma_ax_minus_b = gamma * ax_minus_b
    vk = -gamma_ax_minus_b
    # If number of input arguments is 7 then do CP without linesearch

    if method == "without_linesearch":
        x, y = xy_update(x, y, a, a_t_csr, c, gamma_ax_minus_b, tau)
        alpha = 1

    elif method == "standard":

        alpha, satisfied, x_update2 = initialization(alpha_max, x)

        x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
            alpha_0, x, y, a, a_t_csr, b, c, gamma, gamma_ax_minus_b, tau, x_update2
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                a,
                a_t_csr,
                b,
                c,
                gamma,
                gamma_ax_minus_b,
                tau,
                q,
                x_temp_next,
                y_temp_next,
                x_temp,
                y_temp,
                eps,
                stage,
            )

        x, y = xy_update(x, y, a, a_t_csr, c, gamma_ax_minus_b, tau, alpha)

    elif method == "xyseparate":

        alpha, satisfied, x_update2 = initialization(alpha_max, x)

        x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
            alpha_0, x, y, a, a_t_csr, b, c, gamma, gamma_ax_minus_b, tau, x_update2
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                a,
                a_t_csr,
                b,
                c,
                gamma,
                gamma_ax_minus_b,
                tau,
                q,
                x_temp_next,
                y_temp_next,
                x_temp,
                y_temp,
                eps,
                1,
            )

        y = y + alpha * (-gamma_ax_minus_b)
        alpha, satisfied, _ = initialization(alpha_max, x)

        x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
            alpha_0,
            x,
            y,
            a,
            a_t_csr,
            b,
            c,
            gamma,
            gamma_ax_minus_b,
            tau,
            x_update2,
            y_temp,
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                a,
                a_t_csr,
                b,
                c,
                gamma,
                gamma_ax_minus_b,
                tau,
                q,
                x_temp_next,
                y_temp_next,
                x_temp,
                y_temp,
                eps,
                2,
            )

        x_update1 = x + alpha * tau * (a_t_csr * (y - 2 * gamma_ax_minus_b) - c)
        x_update2 = np.zeros((len(x)))
        x = np.maximum(x_update1, x_update2)
    else:
        raise BaseException(f"unknown method {method}")

    return x, y, vk, num_activate, alpha


def initialization(alpha_max, x):
    alpha = alpha_max
    satisfied = 0
    x_update2 = np.zeros((len(x)))
    return alpha, satisfied, x_update2


def xy_update(x, y, a, a_t_csr, c, gamma_ax_minus_b, tau, alpha=1):
    y = y + alpha * (-gamma_ax_minus_b)
    x_update1 = x + tau * (a_t_csr * (y - 2 * gamma_ax_minus_b) - c)
    x = np.maximum(x_update1, 0)
    return x, y


def pre_alpha_calculation(
    alpha_0,
    x,
    y,
    a,
    a_t_csr,
    b,
    c,
    gamma,
    gamma_ax_minus_b,
    tau,
    x_update2,
    y_temp=None,
):

    if y_temp is None:
        y_temp = y + alpha_0 * (-gamma_ax_minus_b)

    x_update1 = x + tau * (a_t_csr * (y - 2 * gamma_ax_minus_b) - c)
    x_temp = np.maximum(x_update1, x_update2)

    ax_temp_b = a * x_temp - b
    y_temp = y - gamma * ax_temp_b
    x_update1 = x_temp + tau * (a_t_csr * (y_temp - 2 * gamma * ax_temp_b) - c)
    x_temp_next = np.maximum(x_update1, x_update2)
    ax_temp_b_next = a * x_temp_next - b
    y_temp_next = y_temp - gamma * ax_temp_b_next

    return x_temp_next, y_temp_next, x_temp, y_temp


def calculate_alpha(
    alpha,
    factor,
    x,
    x_update2,
    y,
    a,
    a_t_csr,
    b,
    c,
    gamma,
    gamma_ax_minus_b,
    tau,
    q,
    x_temp_next,
    y_temp_next,
    x_temp,
    y_temp,
    eps,
    stage,
):
    if stage:
        alpha = alpha * factor

        y_temp_alpha = y + alpha * (-gamma_ax_minus_b)
        x_update1 = x + tau * (a_t_csr * (y_temp_alpha - 2 * gamma_ax_minus_b) - c)
        x_temp_alpha = np.maximum(x_update1, x_update2)

        ax_temp_alpha_b = a * x_temp_alpha - b
        y_temp_next_alpha = y_temp_alpha - gamma * ax_temp_alpha_b
        x_update1 = x_temp_alpha + tau * (
            a_t_csr * (y_temp_next_alpha - 2 * gamma * ax_temp_alpha_b) - c
        )
        x_temp_next_alpha = np.maximum(x_update1, x_update2)

        r_kplus1 = np.hstack(
            ((y_temp_next_alpha - y_temp_alpha), (x_temp_next_alpha - x_temp_alpha))
        )

        r_bark = np.hstack(((y_temp_next - y_temp), (x_temp_next - x_temp)))

        r_kplus1_norm = np.sqrt((q * r_kplus1).dot(r_kplus1))
        r_bark_norm = np.sqrt((q * r_bark).dot(r_bark))
        satisfied = r_kplus1_norm <= (1 - eps) * r_bark_norm  # Q norm

    else:
        alpha = alpha * factor

        x_temp_alpha = x + alpha * (x_temp - x)
        ax_temp_alpha_b = a * x_temp_alpha - b
        y_temp_alpha = y - gamma * ax_temp_alpha_b

        x_update1 = x_temp_alpha + tau * (
            a_t_csr * (y_temp_alpha - 2 * gamma * ax_temp_alpha_b) - c
        )
        x_temp_next_alpha = np.maximum(x_update1, x_update2)
        y_temp_next_alpha = y_temp_alpha - gamma * (a * x_temp_next_alpha - b)

        r_kplus1 = np.hstack(
            ((y_temp_next_alpha - y_temp_alpha), (x_temp_next_alpha - x_temp_alpha))
        )

        r_bark = np.hstack(((y_temp_next - y_temp), (x_temp_next - x_temp)))

        r_kplus1_norm = np.sqrt((q * r_kplus1).dot(r_kplus1))
        r_bark_norm = np.sqrt((q * r_bark).dot(r_bark))
        satisfied = r_kplus1_norm <= (1 - eps) * r_bark_norm
        # Q norm

    return alpha, satisfied
