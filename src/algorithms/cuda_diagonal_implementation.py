from numba import cuda, jit
import numpy as np
import time
from utils import log_alg_start, log_alg_time


@cuda.jit('void(int32[:], int32[:], int32[:,:], int16, int16, int32)')
def levenshtein_kernel(token1, token2, distances, zy, slice, thread_count):
    thread_index = cuda.grid(1)
    if(thread_index >= thread_count): return

    row = zy + thread_index
    col = slice - row
    if row == 0 or col == 0: return

    score = 0 if token1[row - 1] == token2[col - 1] else 1
    distances[row][col] = min(
        distances[row - 1][col - 1] + score,
        distances[row][col - 1] + 1,
        distances[row - 1][col] + 1
    )


@jit('int16(unicode_type, unicode_type)', forceobj = True)
def levenshtein_root(token1: str, token2: str):
    width = len(token1) + 1
    height = len(token2) + 1

    numeric_token_1 = cuda.to_device(np.array([ord(x) for x in token1], dtype=np.int32))
    numeric_token_2 = cuda.to_device(np.array([ord(x) for x in token2], dtype=np.int32))

    distances = np.zeros((width, height), dtype=np.int32)
    
    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    d_distances = cuda.to_device(distances)

    for slice in range(0, width + height - 1):
        zy = 0 if slice < height else slice - height + 1
        zx = 0 if slice < width else slice - width + 1

        sliceCount = slice - zx + 1 - zy
        numBlocks = int((sliceCount + 256 - 1) / 256)
        levenshtein_kernel[numBlocks, 256](numeric_token_1, numeric_token_2, d_distances, zy, slice, sliceCount)

    return d_distances[-1][-1]

levenshtein_root('a', 'b')

def use_diagonal_cuda_algorithm(input_array) -> np.array:
    array_size = len(input_array)

    log_alg_start('a diagonal agorithm with cuda')

    matrix = np.zeros((array_size, array_size))

    start = time.perf_counter()

    matrix = np.fromfunction(
        np.vectorize(lambda i, j: levenshtein_root(input_array[i], input_array[j])), 
        (array_size, array_size), 
        dtype=int
    )

    log_alg_time(time.perf_counter() - start)

    return matrix