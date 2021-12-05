import numpy as np
import time
from Levenshtein import distance
from polyleven import levenshtein

from utils import log_alg_start, log_alg_time

def use_levenshtein_library(input_array) -> np.array:
    array_size = len(input_array)

    # log_alg_start('python-levenshtein library methods')

    # start = time.perf_counter()
    # matrix = np.fromfunction(
    #     np.vectorize(lambda i, j: distance(input_array[i], input_array[j])), 
    #     (array_size, array_size), 
    #     dtype=int
    # )

    # log_alg_time(time.perf_counter() - start)

    log_alg_start('levenshtein library methods')

    start = time.perf_counter()
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: levenshtein(input_array[i], input_array[j])), 
        (array_size, array_size), 
        dtype=int
    )

    log_alg_time(time.perf_counter() - start)

    return matrix