import numpy as np
import time
from polyleven import levenshtein

from utils import log_alg_start, log_alg_time

def use_polyleven_library(input_array) -> np.array:
    array_size = len(input_array)

    log_alg_start('polyleven library methods')

    start = time.perf_counter()
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: levenshtein(input_array[i], input_array[j], 10)), # 10 is the limit of max difference, can be adjusted
        (array_size, array_size), 
        dtype=int
    )

    log_alg_time(time.perf_counter() - start)

    return matrix