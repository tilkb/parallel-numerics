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
    if (rank==0){
        std::cout<<"OK"<<std::endl;
    }
    MPI_Finalize();
    return 0;
}