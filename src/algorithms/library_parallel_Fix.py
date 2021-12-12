import numpy as np
import time
from Levenshtein import distance
from multiprocessing import cpu_count, Pool
from functools import partial

import numexpr as ne

from utils import log_alg_start, log_alg_time

# final = np.empty([0,100], int)

def dist(a, b):
    return distance(a,b)

   
def calculatePartialMatrix(tuples):
    list1 = tuples[0].tolist()
    list2 = tuples[1].tolist()
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: dist(list2[i], list1[j])), 
        (len(list2), len(list1)), 
        dtype=int
    )
    return matrix

   
def use_levenshtein_library_parallel(complete_array) -> np.array: 
    
    array_size = len(complete_array)
    ncpus = cpu_count()
    equal_pieces = np.array_split(complete_array, ncpus)
    
    log_alg_start('python-levenshtein library methods')
    
    final = np.empty([0,array_size], int)

    start = time.perf_counter()    
        
    data = np.array([[complete_array, equal_pieces[0]], [complete_array, equal_pieces[1]], 
                     [complete_array, equal_pieces[2]], [complete_array, equal_pieces[3]],
                     [complete_array, equal_pieces[4]], [complete_array, equal_pieces[5]], 
                     [complete_array, equal_pieces[6]], [complete_array, equal_pieces[7]],])
    
    result_objs = []

    with Pool(6) as p:
        result = np.array(p.map(calculatePartialMatrix, data))
        result_objs.append(result)
        
    for r in result_objs:
        for j in r:
            final = np.append(final, j, axis= 0)
    
    log_alg_time(time.perf_counter() - start)
    

    return final
    




















