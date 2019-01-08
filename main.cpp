#include <ctime>
#include "uBLAS.hpp"

main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int numberOfProcessors, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix m1(1000,1000);
    Matrix m2(1000,1000);
    m1.init_random();
    m2.init_random();  
    if (rank==0){ 
        std::cout<<"---------------------- MATRIX MULTIPLICATION----------------------"<<std::endl;
        clock_t begin1 = clock();
        uBLAS::naive_multiply(&m1,&m2);
        clock_t end1 = clock();
        double elapsed1 = double(end1 - begin1) / CLOCKS_PER_SEC;
        std::cout<<"NAIVE:" <<elapsed1 <<"sec "<<std::endl;

    
        clock_t begin2 = clock();
        uBLAS::cache_multiply(&m1,&m2);
        clock_t end2 = clock();

        double elapsed2 = double(end2 - begin2) / CLOCKS_PER_SEC;
        std::cout<<"OPTIMIZED:"<< elapsed2 <<"sec "<<std::endl;

        clock_t begin3 = clock();
        uBLAS::vectorization_multiply(&m1,&m2);
        clock_t end3 = clock();


        double elapsed3 = double(end3 - begin3) / CLOCKS_PER_SEC;
        std::cout<<"VECTORIZED:"<< elapsed3 <<"sec "<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    clock_t begin4;
    clock_t end4;
    if (rank==0)
        begin4 = clock();
    uBLAS::parallel_multiply(&m1,&m2, numberOfProcessors, rank);
    if (rank==0){
        end4 = clock();
        double elapsed4 = double(end4 - begin4) / CLOCKS_PER_SEC;
        std::cout<<"MULTI-PROC:"<< elapsed4 <<"sec "<<std::endl;
    }
    if (rank==0)
        std::cout<<"---------------------- MATRIX DECOMPOSITION----------------------"<<std::endl;
    Matrix m(1000,1000);
    Matrix L(1000,1000);
    Matrix U(1000,1000);
    m.init_random();
    Matrix m_small(300,300);
    Matrix L_small(300,300);
    Matrix U_small(300,300);

    if (rank==0){ 
        clock_t begin5 = clock();
        uBLAS::naive_LU(&m,&L,&U);
        clock_t end5 = clock();
        double elapsed5 = double(end5 - begin5) / CLOCKS_PER_SEC;
        std::cout<<"NAIVE LU:" <<elapsed5 <<"sec "<<std::endl;

        clock_t begin6 = clock();
        uBLAS::GAXPY_LU(&m,&L,&U);
        clock_t end6 = clock();
        double elapsed6 = double(end6 - begin6) / CLOCKS_PER_SEC;
        std::cout<<"GAXPY LU:" <<elapsed6 <<"sec "<<std::endl;

        
    }
    MPI_Barrier(MPI_COMM_WORLD);
    clock_t begin7 = clock();
    uBLAS::parallel_LU(&m,&L,&U,numberOfProcessors, rank);
    if (rank==0){
        clock_t end7 = clock();
        double elapsed7 = double(end7 - begin7) / CLOCKS_PER_SEC;
        std::cout<<"MULTI-PROC LU:" <<elapsed7 <<"sec "<<std::endl;

        clock_t begin8 = clock();
        uBLAS::QR(&m,&L,&U);
        clock_t end8 = clock();
        double elapsed8 = double(end8 - begin8) / CLOCKS_PER_SEC;
        std::cout<<"GRAM-SCHMIDT QR:" <<elapsed8 <<"sec "<<std::endl;
    }
    clock_t begin9 = clock();
    uBLAS::QR_hausholder(&m_small,&L_small,&U_small,numberOfProcessors, rank);
    if (rank==0){
        clock_t end9 = clock();
        double elapsed9 = double(end9 - begin9) / CLOCKS_PER_SEC;
        std::cout<<"NAIVE HAUSHOLDER(n/3*n/3 matrix size) QR:" <<elapsed9 <<"sec "<<std::endl;
    }

    if (rank==0){
        Matrix A(100,100);
        Matrix b(100,1);
        A.init_SPD();
        b.init_random();
        //push spectral norm down 
        for (int i=0;i<A.row;i++){
            A.set(i,i,A.get(i,i)*10000);
            b.set(i,0,b.get(i,0)*10000);
        } 
        std::cout<<"---------------------- ITERATIVE SOLVERS-----------------"<<std::endl;
        clock_t begin10 = clock();
        Matrix result = uBLAS::jacobi_iteration(&A, &b);
        clock_t end10 = clock();
        double elapsed10 = double(end10 - begin10) / CLOCKS_PER_SEC;
        std::cout<<"JACOBI ITERATION:" <<elapsed10 <<"sec "<<std::endl;
        clock_t begin11 = clock();
        Matrix result2 = uBLAS::gradient_method(&A, &b);
        clock_t end11 = clock();
        double elapsed11 = double(end11 - begin11) / CLOCKS_PER_SEC;
        std::cout<<"GRADIENT METHOD ITERATION:" <<elapsed11 <<"sec "<<std::endl; //slow convergence, because of numerical problems
        clock_t begin12 = clock();
        Matrix result3 = uBLAS::conjugate_gradient_method(&A, &b);
        clock_t end12 = clock();
        double elapsed12 = double(end12 - begin12) / CLOCKS_PER_SEC;
        std::cout<<"CONJUGATE GRADIENT METHOD ITERATION:" <<elapsed12 <<"sec "<<std::endl; 

    }


    MPI_Finalize();
}