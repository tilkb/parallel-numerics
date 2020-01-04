#include <assert.h>
#include "cuBLAS.cu"

//Copied here to avoid the MPI dependency
Matrix naive_multiply(Matrix *a, Matrix *b){
    if (a->column!=b->row){
        throw std::invalid_argument("Dimension mismatch");
    }
    Matrix result = Matrix(a->row, b->column);
    for (int i=0;i < a->row;i++){
        for (int j=0;j< b->column;j++){
            float dot_product = 0.0;
            for (int k=0;k< a->column;k++){
                dot_product += a->get(i,k) * b->get(k,j);
            }
            result.set(i,j,dot_product);
        }
    }
    return result;
}




void test_GPU_multiply(int n){
    Matrix a(n,n);
    Matrix b(n,n);
    a.init_random();
    b.init_random();
    Matrix result1 = naive_multiply(&a,&b);
    Matrix result2 = cuBLAS::matmul(a,b,0);
    assert(result1.equals(&result2));

    Matrix result3 = cuBLAS::matmul(a,b,1);
    assert(result1.equals(&result3));
    
    Matrix result4 = cuBLAS::matmul(a,b,2);
    assert(result1.equals(&result4));
}



int main(int argc, char *argv[]){
    test_GPU_multiply(16);
    std::cout<<"OK"<<std::endl;
    return 0;
}