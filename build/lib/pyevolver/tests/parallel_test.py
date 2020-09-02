from joblib import Parallel, delayed
from math import sqrt
from pytictoc import TicToc


JOBS = 4
CYCLES = 1000


def eval(random_seed):
    # rs = RandomState(random_seed)
    # time.sleep(0.1)
    # return rs.random()
    # return sqrt(i**2)
    return sum([sqrt(i**2) for i in range(10000)])


def run_parallel():
    tictoc = TicToc()
    tictoc.tic()
    result = Parallel(n_jobs=JOBS, prefer=None)(
        delayed(eval)(rs) for rs in range(CYCLES)
    )
    ellapsed = tictoc.tocvalue()
    print("Parallel: {}".format(ellapsed))
    return result


def run_single():
    tictoc = TicToc()
    tictoc.tic()
    result = [eval(rs) for rs in range(CYCLES)]
    ellapsed = tictoc.tocvalue()
    print("Single: {}".format(ellapsed))
    return result


if __name__ == "__main__":
    result_s = run_single()
    result_p = run_parallel()
    assert result_s == result_p
