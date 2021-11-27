import numpy as np
from utils import log_alg_time, log_alg_start
import time

def levenshtein_distance_DP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1), dtype=np.int32)

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
            
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                distances[t1][t2] = 1 + min(
                    distances[t1][t2 - 1],
                    distances[t1 - 1][t2],
                    distances[t1 - 1][t2 - 1]
                )

    return distances[len(token1)][len(token2)]


def use_custom_dp_algorithm(input_array) -> np.array:
    array_size = len(input_array)

    log_alg_start('a dp algorithm optimised')

    start = time.perf_counter()
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: levenshtein_distance_DP(input_array[i], input_array[j])), 
        (array_size, array_size), 
        dtype=int
    )

    log_alg_time(time.perf_counter() - start)

    return matrix