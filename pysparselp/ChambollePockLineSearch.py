import numpy as np

import scipy.sparse as sparse


methods = ("standard", "activation", "xyseparate", "without_linesearch", "truth")


def chambolle_pock_linesearch(
    a,
    b,
    c,
    nmax=20000,
    eps=1e-6,
    tol=1e-6,
    y_sol=None,
    method="standard",
    callback=None,
):
    """ Solve a primal-dual pair min{c'x|Ax=b,x>=0} and max{b'y|A'y<=c} using Chambolle and Pock algorithm.
    A is a matrix. b and c are column vectors. Input x and y are current x
    and y values. gamma and tau are two user defined parameters for CP algorithm
    Return x and y are updated values after one iteration
    translating from matlab code https://github.com/xiamengqi2012/ChambollePockLinesearch
    A first-order primal-dual algorithm with linesearch. Yura Malitsky Thomas Pock. SIAM Journal on Optimization 2018
    https://arxiv.org/pdf/1608.08883.pdf
    """
    gamma = 1
    tau = 1 / (2 * (sparse.linalg.norm(a) ** 2) * gamma)
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
    q11 = sparse.eye(nr) / gamma
    q12 = -a
    q21 = -a.T
    q22 = sparse.eye(nc) / tau
    q = sparse.bmat([[q11, q12], [q21, q22]])

    while err > tol and n_iter < nmax:
        if n_iter % 1000 == 0:
            print(f"iteration {n_iter} error ={err}")

        n_iter += 1
        x = xnew
        y = ynew
        dist_sol = []

        xnew, ynew, vk, num_activate = chambolle_pock_update(
            a, b, c, x, y, gamma, tau, method, alpha_0, alpha_max, factor, q, eps
        )
        values.append(b.T * ynew)
        err = np.sqrt(
            np.linalg.norm(a * xnew - b) ** 2
            + np.linalg.norm(np.maximum(a.T * ynew - c, 0)) ** 2
        )
        if y_sol is not None:
            dist_sol.append(np.mean(abs(y - y_sol)))
        errors.append(err)

        if callback is not None:
            callback(x, n_iter)

    return {
        "x": xnew,
        "y": ynew,
        "values": values,
        "errors": errors,
        "dist_sol": dist_sol,
    }


def chambolle_pock_update(
    a, b, c, x, y, gamma, tau, method, alpha_0, alpha_max, factor, q, eps
):

    assert method in methods

    ax_minus_b = a * x - b
    num_activate = 0
    gamma_ax_minus_b = gamma * ax_minus_b
    vk = -gamma_ax_minus_b
    # If number of input arguments is 7 then do CP without linesearch

    x, y = xy_update(x, y, a, c, gamma_ax_minus_b, tau)

    use_linesearch = method != "without_linesearch"

    if use_linesearch:

        alpha, satisfied, x_update2 = initialization(alpha_max, x)

    if method == "standard":

        x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
            alpha_0, x, y, a, b, c, gamma, gamma_ax_minus_b, tau, x_update2
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                a,
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

        x, y = xy_update(x, y, a, c, gamma_ax_minus_b, tau, alpha)

    elif method == "activation":

        eps_hat = 0.03
        # need to test
        vkplus1 = -gamma_ax_minus_b
        activate = vkplus1.T * vk / (np.linalg.norm(vkplus1) * np.linalg.norm(vk)) > (
            1 - eps_hat
        )
        if activate:
            num_activate = num_activate + 1

            x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
                alpha_0, x, y, a, b, c, gamma, gamma_ax_minus_b, tau, x_update2
            )
            while alpha * factor > alpha_0 and not satisfied:
                alpha, satisfied = calculate_alpha(
                    alpha,
                    factor,
                    x,
                    x_update2,
                    y,
                    a,
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

            x, y = xy_update(x, y, a, c, gamma_ax_minus_b, tau, alpha)
        else:
            x, y = xy_update(x, y, a, c, gamma_ax_minus_b, tau)

        vk = vkplus1

    elif method == "xyseparate":
        x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
            alpha_0, x, y, a, b, c, gamma, gamma_ax_minus_b, tau, x_update2
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                a,
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
            alpha_0, x, y, a, b, c, gamma, gamma_ax_minus_b, tau, x_update2, y_temp
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                alpha,
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

        x_update1 = x + alpha * tau * (a.T * (y - 2 * gamma_ax_minus_b) - c)
        x_update2 = np.zeros((len(x)))
        x = np.maximum(x_update1, x_update2)
    else:
        raise BaseException(f"unknown method {method}")

    return x, y, vk, num_activate


def initialization(alpha_max, x):
    alpha = alpha_max
    satisfied = 0
    x_update2 = np.zeros((len(x)))
    return alpha, satisfied, x_update2


def xy_update(x, y, a, c, gamma_ax_minus_b, tau, alpha=1):
    y = y + alpha * (-gamma_ax_minus_b)
    x_update1 = x + tau * (a.T * (y - 2 * gamma_ax_minus_b) - c)
    x = np.maximum(x_update1, 0)
    return x, y


def pre_alpha_calculation(
    alpha_0, x, y, a, b, c, gamma, gamma_ax_minus_b, tau, x_update2, y_temp=None
):

    if y_temp is None:
        y_temp = y + alpha_0 * (-gamma_ax_minus_b)

    x_update1 = x + tau * (a.T * (y - 2 * gamma_ax_minus_b) - c)
    x_temp = np.maximum(x_update1, x_update2)

    ax_temp_b = a * x_temp - b
    y_temp = y - gamma * ax_temp_b
    x_update1 = x_temp + tau * (a.T * (y_temp - 2 * gamma * ax_temp_b) - c)
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
        x_update1 = x + tau * (a.T * (y_temp_alpha - 2 * gamma_ax_minus_b) - c)
        x_temp_alpha = np.maximum(x_update1, x_update2)

        ax_temp_alpha_b = a * x_temp_alpha - b
        y_temp_next_alpha = y_temp_alpha - gamma * ax_temp_alpha_b
        x_update1 = x_temp_alpha + tau * (
            a.T * (y_temp_next_alpha - 2 * gamma * ax_temp_alpha_b) - c
        )
        x_temp_next_alpha = np.maximum(x_update1, x_update2)

        r_kplus1 = np.hstack(
            ((y_temp_next_alpha - y_temp_alpha), (x_temp_next_alpha - x_temp_alpha))
        )

        r_bark = np.hstack(((y_temp_next - y_temp), (x_temp_next - x_temp)))

        r_kplus1_norm = np.sqrt((q * r_kplus1).T * r_kplus1)
        r_bark_norm = np.sqrt((q * r_bark).T * r_bark)
        satisfied = r_kplus1_norm <= (1 - eps) * r_bark_norm  # Q norm

    else:
        alpha = alpha * factor

        x_temp_alpha = x + alpha * (x_temp - x)
        ax_temp_alpha_b = a * x_temp_alpha - b
        y_temp_alpha = y - gamma * ax_temp_alpha_b

        x_update1 = x_temp_alpha + tau * (
            a.T * (y_temp_alpha - 2 * gamma * ax_temp_alpha_b) - c
        )
        x_temp_next_alpha = np.maximum(x_update1, x_update2)
        y_temp_next_alpha = y_temp_alpha - gamma * (a * x_temp_next_alpha - b)

        r_kplus1 = hstack(
            ((y_temp_next_alpha - y_temp_alpha), (x_temp_next_alpha - x_temp_alpha))
        )

        r_bark = np.hstack(((y_temp_next - y_temp), (x_temp_next - x_temp)))

        r_kplus1_norm = np.sqrt((q * r_kplus1).T * r_kplus1)
        r_bark_norm = np.sqrt((q * r_bark).T * r_bark)
        satisfied = r_kplus1_norm <= (1 - eps) * r_bark_norm
        # Q norm

    return alpha, satisfied
