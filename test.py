# import numpy as np
# from scipy.linalg import svd

# def principal_angles(U_A, U_B):
#     # Compute the matrix of cosines between the subspaces
#     C = np.dot(U_A.T, U_B)
    
#     # Compute the singular values of C
#     sigma = svd(C, compute_uv=False)
    
#     # Calculate the principal angles in radians
#     principal_angles_rad = np.arccos(np.clip(sigma, -1.0, 1.0))
    
#     return principal_angles_rad

# # Example usage:
# # Subspace A: spanned by columns of U_A
# # Subspace B: spanned by columns of U_B
# U_A = np.array([[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 0]])  # Example 3D basis for a 2D subspace
# U_B = np.array([[1, 1, 0], 
#                 [1, -1, 0], 
#                 [0, 2, 0]])  # Example 3D basis for a

# # Calculate principal angles between the two subspaces
# angles = principal_angles(U_A, U_B)

# # Output principal angles as a vector
# print(f"Principal angles (in radians): {angles}")

# =============================================================================

# import numpy as np
# from scipy.linalg import qr

# # Normal vector to the plane
# n = np.array([1, 1, 1])

# # Find a basis for the subspace by solving the equation n . x = 0
# # Create an identity matrix and find vectors orthogonal to n
# A = np.eye(3) - np.outer(n, n) / np.dot(n, n)

# # The columns of A are the vectors that span the subspace
# _, _, basis_vectors = np.linalg.svd(A)

# # Select two basis vectors for the 2D subspace (since it's in 3D space)
# basis = basis_vectors[1:]

# # Orthonormalize the basis using QR decomposition (which is equivalent to Gram-Schmidt)
# Q, _ = qr(basis.T)

# # The orthonormal basis is the columns of Q
# print("Orthonormal basis:")
# print(Q)

# =============================================================================

import numpy as np
from scipy.linalg import svd

def principal_angles(X, Y):
    """
    Compute the principal angles between two subspaces X and Y.
    
    Parameters:
    X : ndarray of shape (n, p)
        A matrix whose columns form an orthonormal basis for subspace X.
    Y : ndarray of shape (n, q)
        A matrix whose columns form an orthonormal basis for subspace Y.
        
    Returns:
    angles : ndarray of shape (min(p, q),)
        The principal angles in radians between the subspaces.
    """
    
    # Compute the QR factorization (optional, but helps ensure orthonormality)
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    
    # Compute the singular values of Qx^H Qy
    C = np.dot(Qx.T.conj(), Qy)
    _, s, _ = svd(C)
    
    # Compute the principal angles
    angles = np.arccos(np.clip(s, -1, 1))  # Clip to avoid numerical issues
    
    return angles

# Example usage:
# # Subspace X (n=5, p=3)
# X = np.random.randn(5, 3)
# # Subspace Y (n=5, q=2)
# Y = np.random.randn(5, 3)
X = np.array([[1, 0], [0, 0], [0, 0]])  # Subspace in the x-axis direction
Y = np.array([[0, 0], [1, 0], [0, 0]])  # Subspace in the y-axis direction

# Compute principal angles
angles = principal_angles(X, Y)

# Convert angles to degrees if needed
angles_degrees = np.degrees(angles)

print("Principal angles (radians):", angles)
print("Principal angles (degrees):", angles_degrees)