import numpy as np
import time
from polyleven import levenshtein

from utils import log_alg_start, log_alg_time

import dask.array as da


def use_levenshtein_library_parallel_dask(input_array) -> np.array:
    log_alg_start('parallel implementation with dask')

    start = time.perf_counter()
    
    ## First Method
    # matrix = da.frompyfunc(levenshtein, 2, 1).outer(input_array, input_array) 
    
    
    distance = da.frompyfunc(levenshtein, 2, 1)
    
    
    # Second Method
    # size = input_array.size
    # a = da.from_array(input_array, chunks=(size))
    # b = a.map_blocks(lambda x: distance(x[:, None], x), dtype=str, chunks=((size,size),))
    # matrix = b.compute()
    
    
    ## Third Method
    w = da.from_array(input_array, chunks=(2500))
    z = da.blockwise(distance.outer, 'ij', w, 'i', w, 'j', dtype='f8')
    # matrix = z.compute(scheduler='processes')
    matrix = z.compute()
    
    
    log_alg_time(time.perf_counter() - start)

    return matrix
