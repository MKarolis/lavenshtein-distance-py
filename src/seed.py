import string
import random
import numpy as np
from Levenshtein import distance

TEST_DATA_SIZE = 1000
DEFAULT_MIN_STR_LENGTH = 5
DEFAULT_MAX_STR_LENGTH = 60
DEFAULT_CHARS_TO_CHOOSE = string.ascii_uppercase + string.digits + string.punctuation + ' '

TEXT_SAMPLE_FILENAME = './validation-data/sample.txt'
DISTANCES_SAMPLE_FILENAME = './validation-data/distances.csv'


def get_random_string(size=None, chars=DEFAULT_CHARS_TO_CHOOSE):
    sz = size if size else random.randint(DEFAULT_MIN_STR_LENGTH, DEFAULT_MAX_STR_LENGTH)
    return ''.join(random.choice(chars) for _ in range(sz))


if __name__ == '__main__':
    print('\nSeeding initiated for {} records'.format(TEST_DATA_SIZE))
    random_strings: np.array = np.array([get_random_string() for _ in range(TEST_DATA_SIZE)])
    np.savetxt(TEXT_SAMPLE_FILENAME, random_strings, fmt='%s')
    
    print('Entries generated, seeding the distance matrix...')
    distance_matrix = np.fromfunction(
        np.vectorize(lambda i, j: distance(random_strings[i], random_strings[j])), 
        (TEST_DATA_SIZE, TEST_DATA_SIZE), 
        dtype=int
    )
    np.savetxt(DISTANCES_SAMPLE_FILENAME, distance_matrix, fmt='%d', delimiter=';')

    print('Success!')