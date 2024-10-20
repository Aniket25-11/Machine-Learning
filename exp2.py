import numpy as np
from scipy import linalg

# 1. Consider a list datatype then reshape it into 2D, 3D matrix using numpy
list_data = [1, 2, 3, 4, 5, 6]

# Reshape into 2D matrix (2 rows, 3 columns)
matrix_2d = np.reshape(list_data, (2, 3))
print("2D Matrix:\n", matrix_2d)

# Reshape into 3D matrix (1 block, 2 rows, 3 columns)
matrix_3d = np.reshape(list_data, (1, 2, 3))
print("3D Matrix:\n", matrix_3d)

# 2. Generate random matrices using numpy
random_matrix_2x2 = np.random.rand(2, 2)
print("Random 2x2 Matrix:\n", random_matrix_2x2)

random_matrix_3x3 = np.random.rand(3, 3)
print("Random 3x3 Matrix:\n", random_matrix_3x3)

# 3. Find the determinant of matrix using scipy
matrix = np.array([[1, 2], [3, 4]])
det = linalg.det(matrix)
print("Determinant:", det)

# 4. Find eigenvalues and eigenvectors of a matrix using scipy
eigvals, eigvecs = linalg.eig(matrix)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)
