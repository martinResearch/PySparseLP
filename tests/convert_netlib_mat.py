"""conversion of netlib mps file to matlab mat file to be able to test matlab lp solvers"""
from pysparselp.netlib import get_problem

import scipy.io

if __name__ == "__main__":

    problems = [
        "AFIRO",
        "SC105",
        "ADLITTLE",
        "SC50B",
        "SC50A",
        "KB2",
        "ADLITTLE",
        "SCAGR7",
        "PEROLD",
        "AGG2",
    ]
    for name in problems:

        LP = get_problem(name)

        scipy.io.savemat(f"{name}.mat", LP)
