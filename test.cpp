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
    //make it symetric positive definite(SPD)
    A.init_SPD();
    Matrix b(n,1);
    b.init_random();
    
    Matrix x0 = uBLAS::richardson_iteration(&A,&b);
    Matrix b0_projected = uBLAS::vectorization_multiply(&A,&x0);
    assert(b0_projected.equals(&b));

    //push spectral norm down
    for (int i=0;i<n;i++){
        A.set(i,i,A.get(i,i)*50);
    }

    Matrix x1 = uBLAS::jacobi_iteration(&A, &b);
    Matrix b_projected = uBLAS::vectorization_multiply(&A,&x1);
    assert(b_projected.equals(&b));

    //numerical problems!!
    if (n<10){
        Matrix x2 = uBLAS::gradient_method(&A, &b);
        Matrix b_projected2 = uBLAS::vectorization_multiply(&A,&x2);
        assert(b_projected2.equals(&b));
    }

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

bool test_single_eig(Matrix* A, Matrix* x, float eig_val){
    Matrix left_side = uBLAS::vectorization_multiply(A,x);
    x->const_multiply(eig_val);
    return left_side.equals(x,0.1);
}

void test_all_eig(Matrix* A, Matrix* eig_vals, Matrix* eig_vectors){
    for (int i=0;i<A->row;i++){
        //copy eig vec from eig matrix
        Matrix eig_vec(A->row,1);
        for (int j=0;j<A->row;j++){
            eig_vec.set(j,0,eig_vectors->get(j,i));
        }
        assert(test_single_eig(A,&eig_vec,eig_vals->get(i,0)));
    }
}

void test_eig(int n,int numberOfProcessors,int rank){
    Matrix m(n,n);
    m.init_random();
    m.set(0,0,m.get(0,0)*50);// make the biggest eigenvalue big --> fast convergence & no oscillation!

    Matrix eig_vec(n,1);
    if (rank==0){
        float max_eig_val = uBLAS::max_eig(&m, &eig_vec);
        assert(test_single_eig(&m,&eig_vec,max_eig_val));
    }

    m.init_SPD();
    Matrix eig_vals = uBLAS::eig_jacobi(&m, 1, 0);
    Matrix eig_vectors = uBLAS::eig_vectors_from_eig_vals(&m,&eig_vals, 1, 0);
    test_all_eig(&m, &eig_vals, &eig_vectors);
    MPI_Barrier(MPI_COMM_WORLD);
    if (n%(numberOfProcessors*2)==0){
        Matrix eig_vals = uBLAS::eig_jacobi(&m, numberOfProcessors, rank);
        Matrix eig_vectors = uBLAS::eig_vectors_from_eig_vals(&m,&eig_vals, numberOfProcessors, rank);
        if (rank==0){
            test_all_eig(&m, &eig_vals, &eig_vectors);
        }
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
        test_iterative_solver(2);
        test_iterative_solver(5); 
        test_iterative_solver(10);
        test_iterative_solver(50);
        test_iterative_solver(100);   
    }
    test_eig(2,numberOfProcessors, rank);
    test_eig(4,numberOfProcessors, rank);//test parallel too
    test_eig(5,numberOfProcessors, rank);

    if (rank==0){
        std::cout<<"OK"<<std::endl;
    }
    MPI_Finalize();
    return 0;
}