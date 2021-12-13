import numpy as np
import time
from Levenshtein import distance
from multiprocessing import Process, cpu_count, Pipe

from utils import log_alg_start, log_alg_time

def dist(a, b):
    return distance(a,b)

    
def calculatePartialMatrix(input_array, x, send_end):
    matrix = np.fromfunction(
        np.vectorize(lambda i, j: dist(x[i], input_array[j])), 
        (len(x), len(input_array)), 
        dtype=int
    )
    send_end.send(matrix)
   
   
def use_levenshtein_library_parallel(complete_array) -> np.array: 
    array_size = len(complete_array)
    ncpus = cpu_count()
    equal_pieces = np.array_split(complete_array, ncpus)
    
    procesess = []
    pipe_list = []
    
    log_alg_start('python-levenshtein library methods')
    start = time.perf_counter()
    
    for x in range(ncpus):
        recv_end, send_end = Pipe(False)
        process = Process(target=calculatePartialMatrix, args=(complete_array, equal_pieces[x], send_end,))
        procesess.append(process)
        pipe_list.append(recv_end)
        process.start()
        
    for p in procesess:
        p.join()
        
    result = [x.recv() for x in pipe_list]
    final = np.empty([0,array_size], int)
    for r in result:
        final = np.append(final, r, axis= 0)

    log_alg_time(time.perf_counter() - start)

    return final
    




















