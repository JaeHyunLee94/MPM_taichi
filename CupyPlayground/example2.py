import numpy as np
from cupyx.scipy import sparse as cpsp
import scipy
import scipy.sparse
import scipy.sparse.linalg

# Construct a random sparse but regular matrix A
A = scipy.sparse.rand(50,50,density=0.1)+scipy.sparse.coo_matrix(np.diag(np.random.rand(50,)))

# Construct a random sparse right hand side
b = scipy.sparse.rand(50,1,density=0.1).tocsc()

ud = scipy.sparse.linalg.spsolve(A,b) # Works as indented
print(ud)

