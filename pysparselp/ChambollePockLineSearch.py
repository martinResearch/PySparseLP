import numpy as np

import scipy.sparse


methods = ("standard", "activation", "xyseparate", "without_linesearch", "truth")


def ChambollePock(A, b, c, nmax=20000, eps=1e-6, tol=1e-6, y_sol=None):
    """ Solve a primal-dual pair min{c'x|Ax=b,x>=0} and max{b'y|A'y<=c} using Chambolle and Pock algorithm.
    A is a matrix. b and c are column vectors. Input x and y are current x
    and y values. gamma and tau are two user defined parameters for CP algorithm
    Return x and y are updated values after one iteration
    translating from matlab code https://github.com/xiamengqi2012/ChambollePockLinesearch
    """
    gamma = 1
    tau = 1 / (2 * (np.linalg.norm(A) ** 2) * gamma)
    # Restriction: gamma*tau < 1/norm(A)^2

    err = np.inf
    values = []
    err = []

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
    Q11 = sparse.eye(nr) / gamma
    Q12 = -A
    Q21 = -A.T
    Q22 = sparse.eye(nc) / tau
    Q = scipy.sparse.bmat([[Q11, Q12], [Q21, Q22]])
    n = length(c)

    while err > tol and n_iter < nmax:
        if n_iter % 1000 == 0:
            print(f"iteration {n_iter} error ={err}")

        n_iter += 1
        x = xnew
        y = ynew
        dist_sol = []

        xnew, ynew, vk, num_activate = ChambollePockUpdate(
            A, b, c, x, y, gamma, tau, method, alpha_0, alpha_max, factor, Q, eps
        )
        values.append(b.T * ynew)
        err = sqrt(np.linalg.norm(A * xnew - b) ** 2 + np.linalg.norm(np.maximum(A.T * ynew - c, 0)) ** 2)
        if y_sol is not None:
            dist_sol.append(mean(abs(y - y_sol)))
        errors.append(err)

    return {
        "x": xnew,
        "y": ynew,
        "values": values,
        "errors": errors,
        "dist_sol": dist_sol,
    }


def ChambollePockUpdate(
    A, b, c, x, y, gamma, tau, method, alpha_0, alpha_max, factor, Q, vk, eps
):
   

    assert method in methods

    Ax_minus_b = A * x - b
    num_activate = 0
    gamma_Ax_minus_b = gamma * Ax_minus_b
    vk = -gamma_Ax_minus_b
    # If number of input arguments is 7 then do CP without linesearch

    x, y = xy_update(x, y, A, c, gamma_Ax_minus_b, tau)

    use_linesearch = method != "without_linesearch"

    if use_linesearch:

        alpha, satisfied, x_update2 = initialization(alpha_max, x)

    if method == "standard":

        x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
            alpha_0, x, y, A, b, c, gamma, gamma_Ax_minus_b, tau, x_update2
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                A,
                b,
                c,
                gamma,
                gamma_Ax_minus_b,
                tau,
                Q,
                x_temp_next,
                y_temp_next,
                x_temp,
                y_temp,
                eps,
                1,
            )

        x, y = xy_update(x, y, A, c, gamma_Ax_minus_b, tau, alpha)

    elif method == "activation":

        eps_hat = 0.03
        # need to test
        vkplus1 = -gamma_Ax_minus_b
        activate = vkplus1.T * vk / (norm(vkplus1) * norm(vk)) > (1 - eps_hat)
        if activate:
            num_activate = num_activate + 1

            x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
                alpha_0, x, y, A, b, c, gamma, gamma_Ax_minus_b, tau, x_update2
            )
            while alpha * factor > alpha_0 and not satisfied:
                alpha, satisfied = calculate_alpha(
                    alpha,
                    factor,
                    x,
                    x_update2,
                    y,
                    A,
                    b,
                    c,
                    gamma,
                    gamma_Ax_minus_b,
                    tau,
                    Q,
                    x_temp_next,
                    y_temp_next,
                    x_temp,
                    y_temp,
                    eps,
                    1,
                )

            x, y = xy_update(x, y, A, c, gamma_Ax_minus_b, tau, alpha)
        else:
            x, y = xy_update(x, y, A, c, gamma_Ax_minus_b, tau)

        vk = vkplus1

    elif method == "xyseparate":
        x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
            alpha_0, x, y, A, b, c, gamma, gamma_Ax_minus_b, tau, x_update2
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                A,
                b,
                c,
                gamma,
                gamma_Ax_minus_b,
                tau,
                Q,
                x_temp_next,
                y_temp_next,
                x_temp,
                y_temp,
                eps,
                1,
            )

        y = y + alpha * (-gamma_Ax_minus_b)
        alpha, satisfied, _ = initialization(alpha_max, x)

        x_temp_next, y_temp_next, x_temp, y_temp = pre_alpha_calculation(
            alpha_0, x, y, A, b, c, gamma, gamma_Ax_minus_b, tau, x_update2, y_temp
        )
        while alpha * factor > alpha_0 and not satisfied:
            alpha, satisfied = calculate_alpha(
                alpha,
                factor,
                x,
                x_update2,
                y,
                A,
                b,
                c,
                gamma,
                gamma_Ax_minus_b,
                tau,
                Q,
                x_temp_next,
                y_temp_next,
                x_temp,
                y_temp,
                eps,
                2,
            )

        x_update1 = x + alpha * tau * (A.T * (y - 2 * gamma_Ax_minus_b) - c)
        x_update2 = zeros(length(x), 1)
        x = np.maximum(x_update1, x_update2)
    else:
        raise BaseException(f"unknown method {method}")

    return x, y, vk, num_activate


def initialization(alpha_max, x):
    alpha = alpha_max
    satisfied = 0
    x_update2 = zeros(length(x), 1)
    return alpha, satisfied, x_update2


def xy_update(x, y, A, c, gamma_Ax_minus_b, tau, alpha=1):
    y = y + alpha * (-gamma_Ax_minus_b)
    x_update1 = x + tau * (A.T * (y - 2 * gamma_Ax_minus_b) - c)
    x_update2 = zeros(length(x), 1)
    x = np.maximum(x_update1, x_update2)
    return x, y


def pre_alpha_calculation(
    alpha_0, x, y, A, b, c, gamma, gamma_Ax_minus_b, tau, x_update2, y_temp=None
):

    if y_temp is None:
        y_temp = y + alpha_0 * (-gamma_Ax_minus_b)

    x_update1 = x + tau * (A.T * (y - 2 * gamma_Ax_minus_b) - c)
    x_temp = np.maximum(x_update1, x_update2)

    Ax_temp_b = A * x_temp - b
    y_temp = y - gamma * Ax_temp_b
    x_update1 = x_temp + tau * (A.T * (y_temp - 2 * gamma * Ax_temp_b) - c)
    x_temp_next = np.maximum(x_update1, x_update2)
    Ax_temp_b_next = A * x_temp_next - b
    y_temp_next = y_temp - gamma * Ax_temp_b_next

    return x_temp_next, y_temp_next, x_temp, y_temp


def calculate_alpha(
    alpha,
    factor,
    x,
    x_update2,
    y,
    A,
    b,
    c,
    gamma,
    gamma_Ax_minus_b,
    tau,
    Q,
    x_temp_next,
    y_temp_next,
    x_temp,
    y_temp,
    eps,
    stage,
):
    if stage:
        alpha = alpha * factor

        y_temp_alpha = y + alpha * (-gamma_Ax_minus_b)
        x_update1 = x + tau * (A.T * (y_temp_alpha - 2 * gamma_Ax_minus_b) - c)
        x_temp_alpha = np.maximum(x_update1, x_update2)

        Ax_temp_alpha_b = A * x_temp_alpha - b
        y_temp_next_alpha = y_temp_alpha - gamma * Ax_temp_alpha_b
        x_update1 = x_temp_alpha + tau * (
            A.T * (y_temp_next_alpha - 2 * gamma * Ax_temp_alpha_b) - c
        )
        x_temp_next_alpha = np.maximum(x_update1, x_update2)

        r_kplus1 = np.row_stack(
            (y_temp_next_alpha - y_temp_alpha), (x_temp_next_alpha - x_temp_alpha)
        )

        r_bark = np.row_stack((y_temp_next - y_temp), (x_temp_next - x_temp))

        r_kplus1_norm = sqrt((Q * r_kplus1).T * r_kplus1)
        r_bark_norm = sqrt((Q * r_bark).T * r_bark)
        satisfied = r_kplus1_norm <= (1 - eps) * r_bark_norm  # Q norm

    else:
        alpha = alpha * factor

        x_temp_alpha = x + alpha * (x_temp - x)
        Ax_temp_alpha_b = A * x_temp_alpha - b
        y_temp_alpha = y - gamma * Ax_temp_alpha_b

        x_update1 = x_temp_alpha + tau * (
            A.T * (y_temp_alpha - 2 * gamma * Ax_temp_alpha_b) - c
        )
        x_temp_next_alpha = np.maximum(x_update1, x_update2)
        y_temp_next_alpha = y_temp_alpha - gamma * (A * x_temp_next_alpha - b)

        r_kplus1 = np.row_stack(
            (y_temp_next_alpha - y_temp_alpha), (x_temp_next_alpha - x_temp_alpha)
        )

        r_bark = np.row_stack((y_temp_next - y_temp), (x_temp_next - x_temp))

        r_kplus1_norm = sqrt((Q * r_kplus1).T * r_kplus1)
        r_bark_norm = sqrt((Q * r_bark).T * r_bark)
        satisfied = r_kplus1_norm <= (1 - eps) * r_bark_norm
        # Q norm

    return alpha, satisfied
