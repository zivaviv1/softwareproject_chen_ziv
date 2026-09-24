import sys
import numpy as np
import symnmfmodule

np.random.seed(1234)

MAX_ITER = 300
EPSILON = 1e-4

class InputError(Exception):
    pass

def load_points(path):
    return np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)

def initialize_h(w, k):
    mean = float(np.mean(w))
    upper = 2 * np.sqrt(mean / k)
    return np.random.uniform(0, upper, size=(len(w), k))

def print_matrix(matrix):
    for row in matrix:
        print(",".join(f"{float(value):.4f}" for value in row))

def parse_args(args):
    if len(args) != 4 or args[2] not in {"symnmf", "sym", "ddg", "norm"}:
        raise InputError("An Error Has Occurred")
    if not args[1].isdigit() or int(args[1]) <= 1:
        raise InputError("Incorrect number of clusters!")
    k = int(args[1])
    return k, args[2], args[3]

def run(k, goal, path):
    points = load_points(path)
    if k >= len(points):
        raise InputError("Incorrect number of clusters!")
    values = points.tolist()
    if goal == "sym":
        return symnmfmodule.sym(values)
    if goal == "ddg":
        return symnmfmodule.ddg(values)
    w = symnmfmodule.norm(values)
    if goal == "norm":
        return w
    h = initialize_h(w, k)
    return symnmfmodule.symnmf(h.tolist(), w, MAX_ITER, EPSILON)

def main():
    try:
        k, goal, file_name = parse_args(sys.argv)
        print_matrix(run(k, goal, file_name))
    except InputError as error:
        print(error)
        sys.exit(1)
    except (OSError, TypeError, ValueError, RuntimeError, MemoryError):
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()