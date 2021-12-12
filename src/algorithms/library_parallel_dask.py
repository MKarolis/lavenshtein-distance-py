import numpy as np
import time
# from polyleven import levenshtein

from Levenshtein import distance
from multiprocessing import cpu_count

from utils import log_alg_start, log_alg_time

import dask.array as da
import dask
import dask.bag as db

# @dask.delayed
def fuc(a,b):
    c = da.frompyfunc(distance, 2, 1).outer(a, b)
    return c

def use_levenshtein_library_parallel_dask(input_array) -> np.array:
    log_alg_start('python-levenshtein with dask library methods')

    start = time.perf_counter()
    
    array_size = len(input_array)
    ncpus = cpu_count()
    # equal_pieces = np.array_split(input_array, ncpus)
    equal_pieces = db.from_sequence(input_array, npartitions=ncpus)
    
    pieces_matrix = []
    for piece in equal_pieces:
        result = dask.delayed(fuc)(input_array, piece)
        pieces_matrix.append(result)    
       
    final = np.empty([0,array_size], int)
    
    total = dask.delayed(np.append)(final, pieces_matrix, axis=0)
    # print(total.compute())
    
    # matrix = da.frompyfunc(distance, 2, 1).outer(input_array, input_array)
    
    
    log_alg_time(time.perf_counter() - start)
    
    return total.compute()
    # return matrix
