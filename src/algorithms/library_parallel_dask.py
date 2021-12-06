import numpy as np
import time
from Levenshtein import distance

from utils import log_alg_start, log_alg_time

from dask.distributed import Client, progress
import dask.array as da


def use_levenshtein_library_parallel_dask(input_array) -> np.array:
    array_size = len(input_array)

    log_alg_start('python-levenshtein library methods')

    start = time.perf_counter()
    

    arr2 = np.array(['ola','meu','amor'], dtype=str)
    print(da.frompyfunc(distance, 2, 1).outer(arr2, arr2))

    matrix = da.frompyfunc(distance, 2, 1).outer(input_array, input_array)
    
    
    log_alg_time(time.perf_counter() - start)


    return matrix
