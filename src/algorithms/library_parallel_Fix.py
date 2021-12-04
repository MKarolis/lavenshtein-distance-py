import numpy as np
import time
from Levenshtein import distance
from multiprocessing import Process, cpu_count, Pipe, Pool
from functools import partial

import numexpr as ne

from utils import log_alg_start, log_alg_time

# final = np.empty([0,100], int)

def dist(a, b):
    return distance(a,b)

    
# =============================================================================
# def calculatePartialMatrix(input_array, x, send_end):
#     matrix = np.fromfunction(
#         np.vectorize(lambda i, j: dist(x[i], input_array[j])), 
#         (len(x), len(input_array)), 
#         dtype=int
#     )
#     send_end.send(matrix)
# =============================================================================
   
def calculatePartialMatrix(tuples):
    list1 = tuples[0].tolist()
    list2 = tuples[1].tolist()
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: dist(list2[i], list1[j])), 
        (len(list2), len(list1)), 
        dtype=int
    )
    return matrix


def f(array):
    print('hello')
    
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: distance(array[i], array[j])), 
        (len(array), len(array)), 
        dtype=int
    )
    return matrix

   
def use_levenshtein_library_parallel(complete_array) -> np.array: 
    
    array_size = len(complete_array)
    ncpus = cpu_count()
    equal_pieces = np.array_split(complete_array, ncpus)
    # equal_pieces = np.array_split(complete_array, 4)
    
    # procesess = []
    # pipe_list = []
    
    log_alg_start('python-levenshtein library methods')
    
    
# =============================================================================
#     for x in range(ncpus):
#         recv_end, send_end = Pipe(False)
#         process = Process(target=calculatePartialMatrix, args=(complete_array, equal_pieces[x], send_end,))
#         procesess.append(process)
#         pipe_list.append(recv_end)
#         process.start()
#         
#     for p in procesess:
#         p.join()
#         
#     result = [x.recv() for x in pipe_list]
#     final = np.empty([0,array_size], int)
#     for r in result:
#         final = np.append(final, r, axis= 0)
#         # print()
#         # print(r)
# =============================================================================

    final = np.empty([0,array_size], int)
    # datasets = np.empty([ncpu,1], int)
        # for r in result:
        #     final = np.append(final, r, axis= 0)
        #     # print()
        #     # print(r)

    start = time.perf_counter()
    # print()
    # print(final)
    
        
    data = np.array([[complete_array, equal_pieces[0]], [complete_array, equal_pieces[1]], 
                     [complete_array, equal_pieces[2]], [complete_array, equal_pieces[3]],
                     [complete_array, equal_pieces[4]], [complete_array, equal_pieces[5]], 
                     [complete_array, equal_pieces[6]], [complete_array, equal_pieces[7]],])
    # data = np.array([equal_pieces[0], equal_pieces[1], equal_pieces[2], equal_pieces[3] ])
    
    # print(data)
    
    result_objs = []

    with Pool(6) as p:
        result = np.array(p.map(calculatePartialMatrix, data))
        result_objs.append(result)
        # print()
        # print(result)
        # for r in result_obj:
        #     final = np.append(final, r, axis= 0)
        
        # results = [len(result) for result in result_objs]
        # [np.append(final,result, axis= 1) for result in result_objs]
        
    for r in result_objs:
        for j in r:
            # print(r)
            final = np.append(final, j, axis= 0)
    
    print(final)
    
    
    # results = [result.get() for result in result_objs]
        
    
    log_alg_time(time.perf_counter() - start)

    
    # start = time.perf_counter()
    # matrix = np.fromfunction(
    #     np.vectorize(lambda i, j: distance(complete_array[i], complete_array[j])), 
    #     (array_size, array_size), 
    #     dtype=int
    # )
    # log_alg_time(time.perf_counter() - start)
    

    return final
    




















