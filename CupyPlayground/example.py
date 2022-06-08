import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

# Construct a random sparse but regular matrix A
A = scipy.sparse.rand(50, 50, density=0.1) + scipy.sparse.coo_matrix(np.diag(np.random.rand(50, )))

# Construct a random sparse right hand side
b = scipy.sparse.rand(50, 1, density=0.1).tocsc()
b2 = np.random.rand(50, 1)

ud = scipy.sparse.linalg.spsolve(A, b2)  # Works as indented
print(ud)
