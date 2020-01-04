#define BLOCK_SIZE 16
#define CACHE_SIZE 128

#include "matrix.h"



__global__ void row_matmul(float* a, float* b, float* res, int a_row, int a_col, int b_col){
    int x =blockIdx.x;
    __shared__ float column[1024];
    float sum;
    for (int y=threadIdx.x;y<b_col;y+=32){
        sum=0.0f;
        for (int start=0;start<a_col;start+=1024){
            //copy column to local
            for(int i=threadIdx.x;i+start<a_col && i<1024;i+=32){
                column[i] = a[x*a_col+start+i];
            }
            for(int i=0;start+i<a_col && i<1024;i++){
                sum+= column[i]*b[y+i*b_col];
            }
        }
        res[x*b_col+y] = sum;
    }
}

__global__ void element_matmul(float* a, float* b, float* res, int a_row, int a_col, int b_col){
    int x =threadIdx.x+blockIdx.x*blockDim.x;
    int y = threadIdx.y+blockIdx.y*blockDim.y;
    float sum;
    for (int i=0;i<a_col;i++){
        sum += a[x*a_col+i]*b[i*a_col+y];
    }
    res[x*a_col+y] = sum;
}


__global__ void block_matmul(float* a, float* b, float* res, int a_row, int a_col, int b_col){
    int x =threadIdx.x;
    int y =threadIdx.y;
    __shared__ float local_a[CACHE_SIZE][BLOCK_SIZE];
    __shared__ float local_b[BLOCK_SIZE][CACHE_SIZE];
    float sum=0;
    for (int start=0;start<a_col;start+=CACHE_SIZE){
        //load the cached values
        //load row continious from A matrix
        for (int i=0;i<BLOCK_SIZE;i++){
            for (int j=x*BLOCK_SIZE+y;j<CACHE_SIZE && j+start<a_col;j+=(BLOCK_SIZE*BLOCK_SIZE)){
                local_a[j][i] = a[a_col*(i+blockIdx.x*BLOCK_SIZE)+start+j];
            }
        }
        //load row continious from B matrix
        for (int i=0;i<CACHE_SIZE && i+start<b_col;i+=BLOCK_SIZE){
            local_b[x][i+y] = b[(start+i+y)*b_col+x+blockIdx.y*BLOCK_SIZE];
        }

        //calculate
        __syncthreads();
        for (int i=0;i<CACHE_SIZE;i++){
            sum+=local_a[i][x]*local_b[y][i];
        }
    }
    int pos =(blockIdx.x*BLOCK_SIZE+x)*b_col+(blockIdx.y*BLOCK_SIZE+y);
    res[pos] =sum;
}

class cuBLAS{
    public:
        static Matrix matmul_block(Matrix &a, Matrix &b){
            int block_work = 3;
            dim3 blockwidth(BLOCK_SIZE,BLOCK_SIZE);
            int start_pos = sizeof(float) * (BLOCK_SIZE*block_work * a.column);
            cudaHostRegister(a.data,start_pos,0);
            cudaHostRegister(a.data+(BLOCK_SIZE*block_work * a.column),sizeof(float) * ((a.row-BLOCK_SIZE*block_work) * a.column),0);
            //cudaHostRegister(b.data,sizeof(float) * (b.row * b.column), 0);
            float* arr_a = 0;
            float* arr_b = 0;
            float* arr_res = 0;
            cudaMalloc((void **) &arr_a, sizeof(float) * (a.row * a.column));
            cudaMalloc((void **) &arr_b, sizeof(float) * (b.row * b.column));
            cudaMalloc((void **) &arr_res, sizeof(float) * (a.row * b.column));

            cudaStream_t load_stream;
            cudaStream_t calculate_stream;
            cudaStream_t deload_stream;
            cudaStreamCreate ( &load_stream) ;
            cudaStreamCreate ( &calculate_stream) ;
            cudaStreamCreate ( &deload_stream) ;

            //Calculate first row elements
            cudaMemcpy(arr_b, b.data, sizeof(float) * (b.row * b.column), cudaMemcpyHostToDevice);
            cudaMemcpyAsync(arr_a, a.data, sizeof(float) * (BLOCK_SIZE*block_work * a.column), cudaMemcpyHostToDevice,load_stream);
            dim3 first_row_blocknr(block_work, b.column/BLOCK_SIZE);
            cudaDeviceSynchronize();
            block_matmul<<<first_row_blocknr, blockwidth,0, load_stream>>>(arr_a,arr_b,arr_res, a.row,a.column,b.row);
            
            //calculate middle elements
            cudaMemcpyAsync(arr_a+(BLOCK_SIZE*block_work * a.column), a.data+(BLOCK_SIZE*block_work * a.column), sizeof(float) * ((a.row-BLOCK_SIZE*block_work) * a.column), cudaMemcpyHostToDevice, calculate_stream);
            dim3 calculate_blocknr(a.row/BLOCK_SIZE-6, b.column/BLOCK_SIZE);
            cudaDeviceSynchronize();
            block_matmul<<<calculate_blocknr, blockwidth,0, calculate_stream>>>(arr_a+(BLOCK_SIZE*block_work * a.column),arr_b,arr_res+(BLOCK_SIZE * b.column), a.row,a.column,b.row);

            Matrix result = Matrix(a.row, b.column);
            cudaHostRegister(result.data,sizeof(float) * (a.row * b.column), cudaHostRegisterDefault);
            cudaDeviceSynchronize();

            //transfer items while calculate last rows
            int limit = ((a.row-BLOCK_SIZE*block_work) * b.column);
            cudaMemcpyAsync((void *)result.data, (void *)arr_res, limit*sizeof(float), cudaMemcpyDeviceToHost,calculate_stream); 
            block_matmul<<<first_row_blocknr, blockwidth, 0, deload_stream>>>(arr_a+((a.row-BLOCK_SIZE*block_work) * a.column),arr_b,arr_res + limit, a.row,a.column,b.row);
            cudaMemcpyAsync((void *)(result.data+limit), (void *)(arr_res+limit), sizeof(float) *  (BLOCK_SIZE*block_work * b.column), cudaMemcpyDeviceToHost, deload_stream); 


            cudaFree(arr_a); 
            cudaFree(arr_b);
            cudaHostUnregister(a.data);
            cudaHostUnregister(b.data);
            cudaDeviceSynchronize();

            cudaStreamDestroy(load_stream);
            cudaStreamDestroy(calculate_stream);
            cudaStreamDestroy(deload_stream);
             
            cudaFree(arr_res);
            cudaHostUnregister(result.data); 
            return result;
        }

        static Matrix matmul(Matrix &a, Matrix &b, int method){
            Matrix result = Matrix(a.row, b.column);
            float* arr_a = 0;
            float* arr_b = 0;
            float* arr_res = 0;
            cudaMalloc((void **) &arr_a, sizeof(float) * (a.row * a.column));
            cudaMemcpy(arr_a, a.data, sizeof(float) * (a.row * a.column), cudaMemcpyHostToDevice);
            cudaMalloc((void **) &arr_b, sizeof(float) * (b.row * b.column));
            cudaMemcpy(arr_b, b.data, sizeof(float) * (b.row * b.column), cudaMemcpyHostToDevice);
            cudaMalloc((void **) &arr_res, sizeof(float) * (a.row * b.column));
            dim3 blockwidth(4,8);
            dim3 blocknr(a.row/4, b.column/8);
            if (method==0){
                element_matmul<<<blocknr, blockwidth>>>(arr_a,arr_b,arr_res, a.row,a.column,b.row);
            }     
            else if (method==1)
                row_matmul<<<b.column, 32>>>(arr_a,arr_b,arr_res, a.row,a.column,b.row);
            else if (method==2){
                dim3 blockwidth(BLOCK_SIZE,BLOCK_SIZE);
                dim3 blocknr(a.row/BLOCK_SIZE, b.column/BLOCK_SIZE);
                block_matmul<<<blocknr, blockwidth>>>(arr_a,arr_b,arr_res, a.row,a.column,b.row);    
            }
                
            cudaMemcpy((void *)result.data, (void *)arr_res, sizeof(float) *  (a.row * b.column), cudaMemcpyDeviceToHost);
            cudaFree(arr_a); 
            cudaFree(arr_b); 
            cudaFree(arr_res); 
            return result;

        }
};