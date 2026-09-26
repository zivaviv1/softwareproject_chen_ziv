import sys
import numpy as np
import symnmfmodule

EPSILON = 1e-4
MAX_ITER = 300

np.random.seed(1234)

class InputError(Exception):
    pass


def _initialize_h(w, k):
    mean = float(np.mean(w))
    upper = 2 * np.sqrt(mean / k)
    return np.random.uniform(0, upper, size=(len(w), k))


def _print_matrix(matrix):
    for row in matrix:
        print(",".join(f"{float(value):.4f}" for value in row))


def parse_args(args):
    if len(args) != 4:
        raise InputError("An Error Has Occurred")

    k = int(args[1]) if args[1].isdigit() else -1
    goal = args[2]
    file_name = args[3]
        
    if goal not in {"symnmf", "sym", "ddg", "norm"} or k == -1 or k <= 1:
        raise InputError("An Error Has Occurred")
    
    return k, goal, file_name


def run_symnmf(k, goal, file_name):
    # Get the datapoints
    points = np.loadtxt(file_name, delimiter=",", dtype=float, ndmin=2)
    if k >= len(points):
            raise InputError("An Error Has Occurred")
    values = points.tolist()

    # Run according to goal
    if goal == "sym":
        return symnmfmodule.sym(values)
    if goal == "ddg":
        return symnmfmodule.ddg(values)
    w = symnmfmodule.norm(values)
    if goal == "norm":
        return w
    h = _initialize_h(w, k)

    return symnmfmodule.symnmf(h.tolist(), w, MAX_ITER, EPSILON)


def main():
    try:
        k, goal, file_name = parse_args(sys.argv)
        matrix = run_symnmf(k, goal, file_name)
        _print_matrix(matrix)
    except InputError as error:
        print(error)
        sys.exit(1)
    except (OSError, TypeError, ValueError, RuntimeError, MemoryError):
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()