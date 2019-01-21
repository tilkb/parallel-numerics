# μBLAS
This "library" was written to understand the implementation of related problems in parallel numerics. The code covers the main topics (except sparse Matrices) of the Parallel Numerics course at TUM. The parallelism is realized with Open MPI. On a single computer it is far from the optimal due to not using the shared memory model, but it scales well with the number of computers.

This implementation handles only float numbers and it is far from the optimal, as it is optimized only for BLAS-3 operation, so any operation other than Matrix-Matrix multiplication can be easily improved by using BLAS-1 and BLAS-2 optimal methods.


## Main topics
* Matrix multiplication: I optimized for BLAS-3
    * Naive implementation, as it is used in Algebra course
    * Memory optimal implementation: the necessary chunks are in sequential order.
    * Use memory optimal with MAVX2 method, which handles 8 floats together.
    * Multiprocess implementation, it is blockwise to minimize the communication and use heuristics to choose block size.
* LU decomposition
    * Gaussian Elimination
    * GAXPY LU: different order for loops-> helps the compiler to optimize the code
    * Parallel GE: blockwise LU decomposition
* QR decomposition
    * Gram-Schmidt method
    * Navie Hausholder, multiply with full Hausholder matrix -> slow O(n^4)
* Iterative linear equation solver
    * Richardson iteration
    * Jacobi iteration
    * Gradient method: only for SPD matrices
    * Conjugated gradient method: only for SPD matrices, provide faster convergence
* Eigenvalue problems
    * Find maximal eigenvalue with its eigenvector.
    * Eigenvalue search with Givens rotation
        * parallel version
    * Eigenvectors from eigenvalues, conjugated gradient method is used to solve the linear equations
        * parallel version 

## Requirements
* installed OpenMPI (>=3.0)
* Intel processor, which can handle 256 bits operations


## Running
To test the implementation:
`make test`
Due to random initialization and iterative non-stationary methods, tests may fail rarely.

Compile and run the main program, which runs the methods with random matrices and measure the performance. The missing parameter describes the number of processing unit.

`make compile
mpirun -np <#ofprocesses> ./main`

## Results

### Matrix multiplication
| Method | Time (s) |
|--------|------|
|Naive |5.17615|
|Memory optimized|3.18421|
|Vectorized operators|0.42464|
|Multi process|0.256438|

### Matrix decomposition

|Method|Time (s)|
|------|----|
|Naive LU decomposition|0.728499|
|Gaxpy sequential LU decomposition|0.715269|
|Multi process LU|0.306526|
|Naive QR decomposition (Gram–Schmidt) | 5.08438 |
|Naive Hausholder QR decomposition (smaller matrix) |16.4703|

### Iterative solvers
|Method|Time (s)|
|------|----|
|Richardson iteration|0.389017|
|Jacobi iteration|0.000273|
|Gradient method|0.145374|
|Conjugate gradient method|0.002271|

### Eigen value problems
|Method|Time (s)|
|------|----|
|Find maximal eigenvalue with eigenvector|0.000307|
|Eigenvalues with Givens rotation|1.46753|
|Parallel running|2.68838|
|Eigenvectors from eigenvalues|1.20246| 
|Parallel running|0.460136|

