import numpy as np
import time
from multiprocessing import cpu_count
from polyleven import levenshtein
from utils import log_alg_start, log_alg_time

import ray

def compute_partial_matrix(x, complete_array):
    return np.fromfunction(
        np.vectorize(lambda i, j: levenshtein(x[i], complete_array[j], 5)), 
        (len(x), len(complete_array)), 
        dtype=int
    )

def use_ray(complete_array) -> np.array:     
    ncpus = cpu_count()
    equal_pieces = np.array_split(complete_array, ncpus)

    print('Initializing ray')
    start = time.perf_counter()

    ray.init(num_cpus=ncpus)
    remote_function = ray.remote(compute_partial_matrix)
    print('Took {:.6f}s'.format(time.perf_counter() - start))

    log_alg_start('polyleven library with ray parallelization')
    start = time.perf_counter()

    final = np.concatenate(
        ray.get([remote_function.remote(equal_pieces[x], complete_array) for x in range(ncpus)]),
        axis=0
    )
        
    log_alg_time(time.perf_counter() - start)
    
    return final