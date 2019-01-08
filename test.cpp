#include <assert.h>
#include "math.h"
#include "uBLAS.hpp"

void test_multiply(int numberOfProcessors, int rank, int row, int col, int k){
    Matrix m1(row,k);
    Matrix m2(k,col);
    m1.init_random();
    m2.init_random();
    Matrix result1 = uBLAS::naive_multiply(&m1,&m2);
    Matrix result2 = uBLAS::cache_multiply(&m1,&m2);
    assert(result1.equals(&result2));
    Matrix result3 = uBLAS::vectorization_multiply(&m1,&m2);
    assert(result1.equals(&result3));
    MPI_Barrier(MPI_COMM_WORLD);
    Matrix result4 = uBLAS::parallel_multiply(&m1,&m2,numberOfProcessors, rank);
    if (rank==0){
        assert(result1.equals(&result4));
    }
}

void test_decomposition(int n, int numberOfProcessors, int rank){
    Matrix m(n,n);
    m.init_random();
    Matrix L(n,n);
    Matrix U(n,n);
    if (rank==0){
        uBLAS::naive_LU(&m, &L, &U);
        Matrix m2 = uBLAS::vectorization_multiply(&L,&U);
        assert(m.equals(&m2));
        m.init_random();
        uBLAS::GAXPY_LU(&m, &L, &U);
        Matrix m3 = uBLAS::vectorization_multiply(&L,&U);
        assert(m.equals(&m3));
        m.init_random();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    uBLAS::parallel_LU(&m, &L, &U, numberOfProcessors, rank);
    if (rank==0){
        Matrix m2 = uBLAS::vectorization_multiply(&L,&U);
        assert(m.equals(&m2));
    }
}

void test_iterative_solver(int n){
    Matrix A(n,n); 
    //push spectral norm down + make it symetric positive definite(SPD)
    A.init_SPD();
    for (int i=0;i<n;i++){
        A.set(i,i,A.get(i,i)*50);
    }
    Matrix b(n,1);
    b.init_random();
    Matrix x = uBLAS::jacobi_iteration(&A, &b);
    Matrix b_projected = uBLAS::vectorization_multiply(&A,&x);
    assert(b_projected.equals(&b));

    Matrix x2 = uBLAS::gradient_method(&A, &b);
    Matrix b_projected2 = uBLAS::vectorization_multiply(&A,&x2);
    assert(b_projected2.equals(&b));

    Matrix x3 = uBLAS::conjugate_gradient_method(&A, &b);
    Matrix b_projected3 = uBLAS::vectorization_multiply(&A,&x3);
    assert(b_projected3.equals(&b));
}
bool is_orthonormal(Matrix* Q){
    Matrix Q_T(Q->row,Q->column);
    Q_T.clone_data(Q);
    Q_T.transpose();
    Matrix QI = uBLAS::vectorization_multiply(&Q_T, Q);
    Matrix I(Q->row,Q->column);
    I.zero();
    for (int i = 0; i<Q->row;i++){
        I.set(i,i,1.0);
    }
    return (I.equals(&QI));
}


void test_QR(int n, int numberOfProcessors, int rank){
    Matrix m(n,n);
    m.init_random();
    Matrix Q(n,n);
    Matrix R(n,n);
    uBLAS::QR(&m, &Q, &R);
    Matrix result = uBLAS::vectorization_multiply(&Q,&R);
    assert(result.equals(&m));
    //check the orthonormal matrix
    assert(is_orthonormal(&Q));


    Q.init_random();
    R.init_random();
    uBLAS::QR_hausholder(&m, &Q, &R,numberOfProcessors, rank);
    if (rank==0){
        Matrix result2 = uBLAS::vectorization_multiply(&Q,&R);
        assert(result2.equals(&m));
        //check the orthonormal matrix
        assert(is_orthonormal(&Q));
    }
}


int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int numberOfProcessors, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    test_multiply(numberOfProcessors, rank,100,100,100);
    test_multiply(numberOfProcessors, rank,41,41,41);
    test_multiply(numberOfProcessors, rank, 3,3,3);
    test_multiply(numberOfProcessors, rank,21,43,21);
    test_decomposition(60,numberOfProcessors, rank);
    test_decomposition(3,numberOfProcessors, rank);
    test_decomposition(48,numberOfProcessors, rank);
    test_QR(1,numberOfProcessors, rank);
    test_QR(2,numberOfProcessors, rank);
    test_QR(20,numberOfProcessors, rank);
    if (rank==0){
        test_iterative_solver(1);
        test_iterative_solver(5); 
        test_iterative_solver(20); 
    }
    if (rank==0){
        std::cout<<"OK"<<std::endl;
    }
    MPI_Finalize();
    return 0;
}