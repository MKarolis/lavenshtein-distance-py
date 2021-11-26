import numpy as np
from utils import log_alg_time, log_alg_start
import time
from numba import jit
from algorithms.dp_implementation import levenshteinDistanceDP

optimizedDp = jit(levenshteinDistanceDP)

def use_custom_dp_algorithm_optimized(input_array) -> np.array:
    array_size = len(input_array)

    log_alg_start('a dp algorithm optimised with numba.jit')

    start = time.perf_counter()
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: optimizedDp(input_array[i], input_array[j])), 
        (array_size, array_size), 
        dtype=int
    )

    log_alg_time(time.perf_counter() - start)

    return matrix