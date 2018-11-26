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
    clock_t begin4 = clock();
    clock_t end4;
    if (rank==0)
        begin4 = clock();
    uBLAS::parallel_multiply(&m1,&m2, numberOfProcessors, rank);
    if (rank==0){
        end4 = clock();
        double elapsed4 = double(end4 - begin4) / CLOCKS_PER_SEC;
        std::cout<<"MULTI-PROC:"<< elapsed4 <<"sec "<<std::endl;
    }

    MPI_Finalize();
}