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


def LP_primalDualCondat(
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
    useStandardForm=False,
):
    # minimizes c.T*x
    # such that
    # Aeq*x=beq
    # b_lower<= Aineq*x<= b_upper
    # lb<=x<=ub
    #
    # method adapted from
    # A Generic Proximal Algorithm for Convex Optimization
    # Application to Total Variation Minimization

    Aineq, bineq = convertToOnesideInequalitySytem(Aineq, b_lower, b_upper)

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

    if useStandardForm and (not Aineq is None):
        c, Aeq, beq, lb, ub, x0 = convertToStandardFormWithBounds(
            c, Aeq, beq, Aineq, bineq, lb, ub, x0
        )
        Aineq = None

    st = solutionStat(c, AeqCSC, beq, AineqCSC, bineq, callbackFunc)

    for i in range(nb_iter):

        # Update the primal variables

        d = c + y_eq * AeqCSR + y_ineq * AineqCSR
        xtilde = x - tau * d

        np.maximum(xtilde, lb, xtilde)
        np.minimum(xtilde, ub, xtilde)
        z = 2 * xtilde - x
        x = rho * tilde + (1 - rho) * x

        if i % 30 == 0:
            st.eval(x, i)
            if (not max_time is None) and st.elapsed > max_time:
                break

        r_eq = (AeqCSC * z) - beq
        r_ineq = (AineqCSC * z) - bineq

        y_eq_tilde = y_eq + sigma * r_eq
        y_eq = rho * y_eq_tilde + (1 - rho) * y_eq
        y_ineq_tilde = y_ineq + sigma * r_ineq
        y_ineq_tilde = np.maximum(y_ineq_tilde, 0)
        y_ineq = rho * y_ineq_tilde + (1 - rho) * y_ineq

    if not best_integer_solution is None:
        best_integer_solution = best_integer_solution[:n]
    return x[:n], best_integer_solution
