import numpy as np
import time
from multiprocessing import cpu_count
from polyleven import levenshtein
from utils import log_alg_start, log_alg_time
from joblib import Parallel, delayed
import sparse

# RAM usage with 12 threads and 100k records:
# 1e7     - 8  GB RAM
# 2 * 1e7 - 16 GB RAM
# 1e8     - 32 GB RAM (maximum capacity)
# How many integers can a chunk hold?
DESIRED_CHUNK_SIZE = 2 * 1e7


def compute_partial_matrix(x, complete_array):
    return sparse.COO.from_numpy(np.fromfunction(
        np.vectorize(lambda i, j: levenshtein(x[i], complete_array[j], 5)), 
        (len(x), len(complete_array)), 
        dtype=int
    ), fill_value=6)

def use_joblib_sparse(complete_array) -> sparse.COO:     
    chunk_count = max(1, int((len(complete_array)**2)/DESIRED_CHUNK_SIZE))
    worker_count = min(cpu_count(), chunk_count)
    equal_pieces = np.array_split(complete_array, chunk_count)

    log_alg_start('polyleven library with joblib parallelization and sparse matrices')
    print('Using {} worker(s) and {} chunks'.format(worker_count, chunk_count))
    start = time.perf_counter()
    final = sparse.concatenate(
        Parallel(n_jobs=worker_count)(delayed(compute_partial_matrix)(equal_pieces[x], complete_array) for x in range(chunk_count)), 
        axis=0
    )
    log_alg_time(time.perf_counter() - start)
    
    print('Memory footprint of generated sparse matrix: {} bytes'.format(final.nbytes))
    return final