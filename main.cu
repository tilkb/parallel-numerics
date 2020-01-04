#include <ctime>
#include "cuBLAS.cu"


int main(int argc, char *argv[]){
    Matrix m1(1024,1024);
    Matrix m2(1024,1024);
    m1.init_random();
    m2.init_random();

    clock_t begin1 = clock();
    cuBLAS::matmul(m1,m2,0);
    clock_t end1 = clock();
    double elapsed1 = double(end1 - begin1) / CLOCKS_PER_SEC;
    std::cout<<"GPU naive all thread used:" <<elapsed1 <<"sec "<<std::endl;

    clock_t begin2 = clock();
    cuBLAS::matmul(m1,m2,1);
    clock_t end2 = clock();
    double elapsed2 = double(end2 - begin2) / CLOCKS_PER_SEC;
    std::cout<<"GPU a row for each block(shared memory used for cache the row of the first matrix's row):" <<elapsed2 <<"sec "<<std::endl;

    clock_t begin3 = clock();
    cuBLAS::matmul(m1,m2,2);
    clock_t end3 = clock();
    double elapsed3 = double(end3 - begin3) / CLOCKS_PER_SEC;
    std::cout<<"block optimized GPU:" <<elapsed3 <<"sec "<<std::endl;

    clock_t begin4 = clock();
    cuBLAS::matmul_block(m1,m2);
    clock_t end4 = clock();
    double elapsed4 = double(end4 - begin4) / CLOCKS_PER_SEC;
    std::cout<<"block optimized GPU with async streaming:" <<elapsed4 <<"sec "<<std::endl;

}