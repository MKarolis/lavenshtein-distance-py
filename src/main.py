from algorithms.library_parallel import use_levenshtein_library

from algorithms.library_parallel import parallel

from algorithms.dp_implementation import use_custom_dp_algorithm
from algorithms.dp_implementation_numba import use_custom_dp_algorithm_optimized
from algorithms.diagonal_implementation import use_diagonal_dp_algorithm_base
#from algorithms.cuda_diagonal_implementation import use_diagonal_cuda_algorithm
from seed import DISTANCES_SAMPLE_FILENAME, TEXT_SAMPLE_FILENAME
import pandas as pd
import csv
import numpy as np
import multiprocessing


def get_distance_matrix(input_array) -> np.array:
    """
    Calculates a matrix of Levenshtein distance for a given input array

    Parameters
    ----------
    input_array: an array of string values.

    Returns
    ----------
    distance_matrix: A 2D array of size NxN (where N is the size of input_array), 
        where distance_matrix[i][j] is the Levenshtein distance from input_array[i] to input_array[j]
    """
    
    # Change the following lines to apply an algorithm of your choice

    # Very fast, 10k - 214s, 1000 - 2s, 100 - under a second
    algorithm = use_levenshtein_library
    # algorithm = parallel

    # Very slow, 100 - 17s
    # algorithm = use_custom_dp_algorithm 

    # Slow, 100 - 2.4s
    # algorithm = use_custom_dp_algorithm_optimized 

    # Slow, 100 - 2.2s
    # algorithm = use_diagonal_dp_algorithm_base

    # Overwhelmingly slow, 100 - 69s (Nice!)
    # algorithm = use_diagonal_cuda_algorithm

    return algorithm(input_array)


def verify_matrix_correctness(input_array, computed_matrix: np.array):
    """
    Verifies that computed distance matrix is correct by comparing it to pre-computed lavenshtein distance results

    Parameters
    ----------
    input_array: an array of string values.
    computed_matrix: distance matrix that needs to be verified
    """

    target_matrix = np.genfromtxt(DISTANCES_SAMPLE_FILENAME, delimiter=";")
    diffs: np.array = np.argwhere(computed_matrix != target_matrix)

    if len(diffs):
        print('Found incorrectly calculated distances:')

        diffs_to_ignore = set()
        for diff in diffs:
            if (diff[1], diff[0]) in diffs_to_ignore:
                continue
            print('Levenshtein distance between {} and {} was {}, should be {}'.format(
                input_array[diff[0]], input_array[diff[1]], computed_matrix[diff[0], diff[1]], target_matrix[diff[0], diff[1]]
            ))
            diffs_to_ignore.add((diff[1], diff[0]))
        
        print('{} errors in total'.format(len(diffs_to_ignore)))
    else:
        print('Distance validation succeeded!')
    


if __name__ == '__main__':
    print()

    input_array = pd.read_csv(
        TEXT_SAMPLE_FILENAME, 
        delimiter='\n', 
        quoting=csv.QUOTE_NONE, 
        comment=None, 
        header=None, 
        dtype=str
    )[0].to_numpy()     

    matrix = get_distance_matrix(input_array)

    print('')
    # verify_matrix_correctness(input_array, matrix)

    print()
    
    