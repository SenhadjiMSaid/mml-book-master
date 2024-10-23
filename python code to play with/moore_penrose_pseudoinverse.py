import numpy as np


def moore_penrose_pseudoinverse(A):
    # Step 1: Perform Singular Value Decomposition
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

    # Step 2: Compute the pseudoinverse of sigma (1/sigma for non-zero elements)
    # Create a diagonal matrix with the reciprocal of the non-zero singular values
    sigma_plus = np.diag([1 / s if s > 1e-10 else 0 for s in sigma])

    # Step 3: Compute the pseudoinverse using the formula A^+ = V * Sigma^+ * U^T
    A_pseudoinverse = Vt.T @ sigma_plus @ U.T

    return A_pseudoinverse


# Example usage:
A = np.array([[0, -1, 2, 0, 2],
              [1, -3, 1, -1, -2],
              [-3, 4, 1, 1, 2],
              [-1, -3, 5, 0, 7]])

A_pseudoinverse = moore_penrose_pseudoinverse(A)
print("Moore-Penrose Pseudoinverse of A:\n", A_pseudoinverse)
