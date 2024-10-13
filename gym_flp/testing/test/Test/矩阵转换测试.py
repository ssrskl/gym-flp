from gym_flp.util.AuxiliaryUtils import transfer_matrix
import numpy as np


def test_transfer_matrix():
    matrix = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
    print(transfer_matrix(matrix))
    print(transfer_matrix(transfer_matrix(matrix)))

test_transfer_matrix()
