import numpy as np
import time
from multiprocessing import cpu_count
from polyleven import levenshtein
from utils import log_alg_start, log_alg_time
from mpire import WorkerPool

def compute_partial_matrix(x, complete_array):
    return np.fromfunction(
        np.vectorize(lambda i, j: levenshtein(x[i], complete_array[j])), 
        (len(x), len(complete_array)), 
        dtype=int
    )

def use_mpire(complete_array) -> np.array:     
    ncpus = cpu_count()
    equal_pieces = np.array_split(complete_array, ncpus)

    log_alg_start('polyleven library with mpire parallelization')
    start = time.perf_counter()

    with WorkerPool(n_jobs=ncpus) as pool:
        final = pool.map(
            compute_partial_matrix, 
            ((equal_pieces[x], complete_array) for x in range(ncpus)), 
            iterable_len=ncpus
        )

    log_alg_time(time.perf_counter() - start)
    
    return final