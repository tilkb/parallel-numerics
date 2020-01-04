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
| Method |Machine1 Time (s) |Machine2 Time (s)
|--------|------|------|
|Naive |5.17615|2.29718|
|Memory optimized|3.18421|1.73342|
|Vectorized operators|0.42464|0.236444|
|Multi process|0.256438|0.102113|

### GPU optimized matrix multiplication 
The code is optimized for Nvidia MX150

| Method |Time (s) |
|--------|------|
|Naive |0.125388|
|ROw in shared memory|0.055226|
|Block optimized|0.036619|
|Block optimized with async data streaming|0.035769|

Async data copy can be seen on this figure:
![alt text](img/async.png "Image from profiler")


### Matrix decomposition

| Method |Machine1 Time (s) |Machine2 Time (s)
|------|----|------|
|Naive LU decomposition|0.728499|0.350948|
|Gaxpy sequential LU decomposition|0.715269|0.341812|
|Multi process LU|0.306526|0.103951|
|Naive QR decomposition (Gram–Schmidt) | 5.08438 |1.86334|
|Naive Hausholder QR decomposition (smaller matrix) |16.4703|17.0201|

### Iterative solvers
| Method |Machine1 Time (s) |Machine2 Time (s)
|------|----|------|
|Richardson iteration|0.389017|0.587187|
|Jacobi iteration|0.000273|0.00049|
|Gradient method|0.145374|0.188615|
|Conjugate gradient method|0.002271|0.004664|

### Eigen value problems
| Method |Machine1 Time (s) |Machine2 Time (s)
|------|----|------|
|Find maximal eigenvalue with eigenvector|0.000307|0.000382|
|Eigenvalues with Givens rotation|1.46753|1.73137|
|Parallel running|2.68838|1.49575|
|Eigenvectors from eigenvalues|1.20246| 2.19577|
|Parallel running|0.460136|0.725736|

