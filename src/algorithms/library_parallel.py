import numpy as np
import time
from Levenshtein import distance
import multiprocessing

from utils import log_alg_start, log_alg_time

def dist(a, b):
    return distance(a,b)

def vectorize(input_array):
    cpus = multiprocessing.cpu_count()
    list = np.array_split(input_array, cpus)
    
    new = []
        
    for x in list:
        matrix = np.fromfunction(
            np.vectorize(lambda i, j: distance(x[i], input_array[j])), 
            (len(x), len(input_array)), 
            dtype=int
        )
        print(matrix)
        print()
        
    return np.vectorize(lambda i, j: dist(input_array[i], input_array[j]))

def use_levenshtein_library(input_array) -> np.array: 
    #parallel(input_array)
    array_size = len(input_array)

    log_alg_start('python-levenshtein library methods')

    start = time.perf_counter()
    matrix = np.fromfunction(
        vectorize(input_array), 
        (array_size, array_size), 
        dtype=int
    )

    log_alg_time(time.perf_counter() - start)

    return matrix



def parallel(input_array):
    cpus = multiprocessing.cpu_count()
    
    array_size = len(input_array)
    #valores = np.random.randint(100, size=50)
    list = np.array_split(input_array, cpus)
    print(len(list))
    
    procesess = []
    
    for i in range(cpus):
        print()
        
        print(list[i])
        
        # print(use_levenshtein_library(list[i]))
        
        # process = multiprocessing.Process(target=use_levenshtein_library, args=(list[i]))
        # procesess.append(process)
        # process.start()
        
    # for process in procesess:
    #     process.join()
        
    
    
    # print(input_array)
    # print(array_size)




















