"""Tests using netlib problems."""

import copy
import json
import os


import matplotlib.pyplot as plt

import numpy as np


from pysparselp.SparseLP import SparseLP, solving_methods
from pysparselp.netlib import getProblem

__folder__ = os.path.dirname(__file__)


def solve_netlib(pbname, display=False, max_time_seconds=30):

    LPDict = getProblem(pbname)
    groundTruth = LPDict["solution"]

    LP = SparseLP()
    nbvar = len(LPDict["costVector"])
    LP.addVariablesArray(
        nbvar,
        lowerbounds=LPDict["lowerbounds"],
        upperbounds=np.minimum(LPDict["upperbounds"], np.max(groundTruth) * 2),
        costs=LPDict["costVector"],
    )
    LP.addEqualityConstraintsSparse(LPDict["Aeq"], LPDict["Beq"])
    LP.addConstraintsSparse(LPDict["Aineq"], LPDict["B_lower"], LPDict["B_upper"])

    print("solving")
    if display:
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].set_title("mean absolute distance to solution")
        axarr[1].set_title("maximum constraint violation")
        axarr[2].set_title("difference with optimum value")

    LP2 = copy.deepcopy(LP)
    LP2.convertToOnesideInequalitySystem()

    # LP2.saveMPS(os.path.join(thisfilepath,'data','example_reexported.mps'))

    LP = LP2
    assert LP.checkSolution(groundTruth)
    costGT = LP.costsvector.dot(groundTruth.T)
    print("gt  cost :%f" % costGT)

    # scipySol,elapsed=LP2.solve(method='ScipyLinProg',getTiming=True,nb_iter=100000)

    # method='ScipyLinProg'
    # if not scipySol is np.nan:
    # sol1=scipySol
    # maxv=LP.maxConstraintViolation(sol1)
    # compute the primal and dual infeasibility
    # print ('%s found  solution with maxviolation=%2.2e and  cost %f (vs %f for ground truth) in %f seconds'%(method,maxv,LP.costsvector.dot(sol1),costGT,elapsed))
    # print ('mean of absolute distance to gt solution =%f'%np.mean(np.abs(groundTruth-sol1)))
    # else:
    # print ('scipy simplex did not find a solution')

    # testing our methods

    solving_methods2 = [m for m in solving_methods if (m not in ["ScipyLinProg"])]
    # solving_methods2=['Mehrotra']
    distanceToGroundTruth = {}
    for method in solving_methods2:
        print(
            "\n\n----------------------------------------------------------\nSolving LP using %s"
            % method
        )
        sol1, elapsed = LP.solve(
            method=method,
            getTiming=True,
            nb_iter=1000000,
            max_time=max_time_seconds,
            groundTruth=groundTruth,
            plotSolution=None,
            nb_iter_plot=500,
        )
        distanceToGroundTruth[method] = LP.distanceToGroundTruth
        if display:
            axarr[0].semilogy(LP.opttime_curve, LP.distanceToGroundTruth, label=method)
            axarr[1].semilogy(LP.opttime_curve, LP.max_violated_constraint)
            axarr[2].semilogy(LP.opttime_curve, LP.pobj_curve - costGT)
            axarr[0].legend()
            plt.show()
    print("done")
    return distanceToGroundTruth


def trim_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


def test_netlib(update_results=False, display=False):
    max_time_seconds = 10
    pb_names = ["SC105"]
    for pb_name in pb_names:
        distanceToGroundTruthCurves = solve_netlib(
            pb_name, display=False, max_time_seconds=max_time_seconds
        )

        curves_json_file = os.path.join(__folder__, f"netlib_curves_{pb_name}.json")
        if update_results:
            with open(curves_json_file, "w") as f:
                json.dump(distanceToGroundTruthCurves, f)

        with open(curves_json_file, "r") as f:
            distanceToGroundTruthCurves_expected = json.load(f)

        for k, v1 in distanceToGroundTruthCurves.items():
            v2 = distanceToGroundTruthCurves_expected[k]
            tv1, tv2 = trim_length(v1, v2)
            max_diff = np.max(np.abs(np.array(tv1) - np.array(tv2)))
            print(f"{pb_name} max diff {k} = {max_diff}")
            np.testing.assert_almost_equal(*trim_length(v1, v2))


if __name__ == "__main__":

    # test_netlib('afiro')# seems like the solution is not unique
    # test_netlib('SC50B')
    # test_netlib('SC50A')
    # test_netlib('KB2')
    test_netlib("SC105", display=False)
    # test_netlib('ADLITTLE')# seems like the solution is not unique
    # test_netlib('SCAGR7')
    # test_netlib('PEROLD')# seems like there is a loading this problem
    # test_netlib('AGG2')
