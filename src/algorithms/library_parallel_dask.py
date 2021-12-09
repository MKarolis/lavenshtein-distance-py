import numpy as np
import time
from polyleven import levenshtein

from utils import log_alg_start, log_alg_time

import dask.array as da


def use_levenshtein_library_parallel_dask(input_array) -> np.array:
    log_alg_start('parallel implementation with dask')

    start = time.perf_counter()
    
    matrix = da.frompyfunc(levenshtein, 2, 1).outer(input_array, input_array) 
    
    log_alg_time(time.perf_counter() - start)

    return matrix
