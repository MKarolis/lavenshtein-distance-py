import numpy as np
from utils import log_alg_time, log_alg_start
import time
from numba import njit

@njit
def diagonal(token1, token2):
    width = len(token1) + 1
    height = len(token2) + 1

    distances = np.zeros((width, height))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    for slice in range(0, width + height - 1):
        zy = 0 if slice < height else slice - height + 1
        zx = 0 if slice < width else slice - width + 1

        for row in range(zy, slice - zx + 1):
            col = slice - row
            if row == 0 or col == 0: continue

            score = 0 if token1[row - 1] == token2[col - 1] else 1
            distances[row][col] = min(
                distances[row - 1][col - 1] + score,
                distances[row][col - 1] + 1,
                distances[row - 1][col] + 1
            )
    return distances[-1][-1]


diagonal('a', 'b')


def use_diagonal_dp_algorithm_base(input_array) -> np.array:
    array_size = len(input_array)

    log_alg_start('a dp algorithm optimised')


    matrix = np.zeros((array_size, array_size))

    start = time.perf_counter()

    matrix = np.fromfunction(
        np.vectorize(lambda i, j: diagonal(input_array[i], input_array[j])), 
        (array_size, array_size), 
        dtype=int
    )

    log_alg_time(time.perf_counter() - start)

    return matrix