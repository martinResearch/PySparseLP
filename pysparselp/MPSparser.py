"""Function to load MPS files."""

import gzip

import numpy as np

from scipy import sparse


def mps_parser(f, fsol=None):
    """
    Parse Linear programs in the MPS format.
    This file format is described here
    https://en.wikipedia.org/wiki/MPS_(format)
    This has been coded in a rush and might not handle all cases
    could have a look at mps2mat.f that is a part of the package LIPSOL by Yin Zhang
    or https://github.com/YimingYAN/cppipm/blob/master/src/mpsReader.cpp
    """
    nb_ineq = 0
    nb_eq = 0
    nb_var = 0
    b_lower = dict()
    b_upper = dict()
    b_eq = dict()
    rows = dict()
    variables = dict()
    a_ineq_list = []
    a_eq_list = []
    v_id_to_var = dict()
    part_parsing = None
    while True:

        line = f.readline()
        line = line.rstrip("\n")
        line += " " * (len(line) - 61)
        t = []
        t.append(line[1:3].strip())
        t.append(line[4:12].ljust(8))
        t.append(line[14:22])
        t.append(line[25:36].strip())
        t.append(line[39:47])
        t.append(line[49:61].strip())
        # t=line.split()
        # t[0]=
        # 2-3        5-12      15-22     25-36     40-47     50-61
        if len(t) == 0:
            continue
        if line.startswith("ENDATA"):
            break
        if line.startswith("*"):  # this is a comment
            continue
        if len(t) == 0:
            continue
        if line.startswith("NAME"):
            problem_name = t[1]
            continue
        if line.startswith("ROWS"):
            part_parsing = "ROWS"
            continue
        if line.startswith("COLUMNS"):
            part_parsing = "COLUMNS"
            continue
        if line.startswith("RHS"):
            part_parsing = "RHS"
            continue
        if line.startswith("BOUNDS"):
            part_parsing = "BOUNDS"
            continue

        if line.startswith("RANGES"):
            print("not coded yet")
            raise

        if part_parsing == "ROWS":
            if t[0] == "N":
                costname = t[1]

            if t[1] in rows:
                raise
            r = dict()
            rows[t[1]] = r
            r["type"] = t[0]
            if t[0] == "G":
                r["id"] = nb_ineq
                b_lower[nb_ineq] = 0
                b_upper[nb_ineq] = np.inf
                nb_ineq += 1

            if t[0] == "L":
                r["id"] = nb_ineq
                b_lower[nb_ineq] = -np.inf
                b_upper[nb_ineq] = 0
                nb_ineq += 1
            elif t[0] == "E":
                r["id"] = nb_eq
                b_eq[nb_eq] = 0  # set default value
                nb_eq += 1

            continue

        if part_parsing == "COLUMNS":

            if t[1] in variables:

                var = variables[t[1]]
            else:
                var = dict()
                variables[t[1]] = var
                var["id"] = nb_var
                var["UP"] = np.inf
                var[
                    "LO"
                ] = 0  # Variables not mentioned in a given BOUNDS set are taken to be non-negative (lower bound zero, no upper bound)
                var["cost"] = 0
                v_id_to_var[nb_var] = var
                nb_var += 1

            j = var["id"]
            for k in range(int((len(t) - 2) / 2)):
                if t[2 * k + 2] == "":
                    break
                r = rows[t[2 * k + 2]]
                v = float(t[2 * k + 3])
                if r["type"] == "N":
                    var["cost"] = v
                    continue

                i = r["id"]

                if r["type"] == "L":
                    a_ineq_list.append((i, j, v))
                elif r["type"] == "G":
                    a_ineq_list.append((i, j, v))
                elif r["type"] == "E":
                    a_eq_list.append((i, j, v))
            continue

        if part_parsing == "RHS":

            for k in range(int((len(t) - 2) / 2)):
                if t[2 * k + 2] == "":
                    break
                r = rows[t[2 * k + 2]]
                i = r["id"]
                v = float(t[2 * k + 3])
                if r["type"] == "N":
                    raise
                elif r["type"] == "L":

                    b_upper[i] = v
                elif r["type"] == "G":
                    b_lower[i] = v

                elif r["type"] == "E":
                    b_eq[i] = v
            continue

        if part_parsing == "BOUNDS":
            var = variables[t[2]]
            var["name"] = t[2]
            if t[0] == "UP" or t[0] == "LO":
                var[t[0]] = float(t[3])
            elif t[0] == "FR":
                var["UP"] = np.inf
                var["LO"] = -np.inf
            elif t[0] == "FX":
                var["UP"] = float(t[3])
                var["LO"] = float(t[3])
            elif t[0] == "MI":
                var["LO"] = -np.inf
            elif t[0] == "PL":
                var["UP"] = np.inf
            elif t[0] == "BV" or t[0] == "LI" or t[0] == "UI":
                print("integer constraints ignored")
                raise

    cost_vector = np.array([v_id_to_var[i]["cost"] for i in range(nb_var)])
    upper_bounds = np.array([v_id_to_var[i]["UP"] for i in range(nb_var)])
    lower_bounds = np.array([v_id_to_var[i]["LO"] for i in range(nb_var)])

    a_ineq = sparse.dok_matrix((nb_ineq, nb_var))
    for i, j, v in a_ineq_list:
        a_ineq[i, j] = v

    a_eq = sparse.dok_matrix((nb_eq, nb_var))
    for i, j, v in a_eq_list:
        a_eq[i, j] = v

    b_eq = np.array([b_eq[i] for i in range(nb_eq)])
    b_lower = np.array([b_lower[i] for i in range(nb_ineq)])
    b_upper = np.array([b_upper[i] for i in range(nb_ineq)])

    # print a_eq
    r = {
        "cost_vector": cost_vector,
        "upper_bounds": upper_bounds,
        "lower_bounds": lower_bounds,
        "a_eq": a_eq,
        "b_eq": b_eq,
        "a_ineq": a_ineq,
        "b_lower": b_lower,
        "b_upper": b_upper,
        "problem_name": problem_name,
        "costname": costname
    }

    # parses Linear Program solution file generated by perPlex Version 1.00
    # examples of such file in http://www.zib.de/koch/perplex/data/netlib/txt/
    # paper here https://opus4.kobv.de/opus4-zib/files/727/ZR-03-05.pdf
    r["solution"] = None
    if fsol is not None:

        while True:

            line = fsol.readline()
            line = line.rstrip("\n")
            if line == "":
                continue
            if len(t) == 0:
                continue
            if line.startswith("- EOF"):
                break

            if line.startswith("* Objvalue"):
                # objvalue = 4
                continue
            if line.startswith("- Variables"):
                part_parsing = "Variables"
                continue

            if line.startswith("- Constraints"):
                part_parsing = "Constraints"
                continue

            if part_parsing == "Variables":
                if line.startswith("V Name"):
                    name = line.split(": ")[1].ljust(8)
                    var = variables[name]
                    continue

                if line.startswith("V Value"):
                    val1 = float(line.split(":")[1].split("=")[0])
                    frac = line.split(":")[1].split("=")[1].split("/")
                    if len(frac) == 1:
                        val = float(frac[0])
                    else:
                        val = float(frac[0]) / float(frac[1])
                    if np.isnan(val):  # happends with PEROLD
                        var["sol"] = val1
                    else:
                        var["sol"] = val
                    continue

                if line.startswith("V State    : on lower"):
                    var["sol"] = var["LO"]
                    continue

                if line.startswith("V State    : on upper"):
                    var["sol"] = var["UP"]
                    continue

                if line.startswith("V State    : on both"):
                    assert var["UP"] == var["LO"]
                    var["sol"] = var["UP"]
                    continue

        solution = np.array([v_id_to_var[i]["sol"] for i in range(nb_var)])

        r["solution"] = solution

    return r


if __name__ == "__main__":

    filename_lp = "./data/netlib/AFIRO.SIF"
    filename_sol = "./data/perPlex/afiro.txt.gz"
    file_lp = open(filename_lp, "r")
    fsol = gzip.open(filename_sol, "r")
    LP = mps_parser(file_lp, fsol)
