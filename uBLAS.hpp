#include <stdexcept>
#include <math.h>
#include <immintrin.h>
#include "mpi.h"
#include "matrix.h"

class uBLAS{
    public:
        //Naive implementation
        static Matrix naive_multiply(Matrix *a, Matrix *b){
            if (a->column!=b->row){
                throw std::invalid_argument("Dimension mismatch");
            }
            Matrix result = Matrix(a->row, b->column);
            for (int i=0;i < a->row;i++){
                for (int j=0;j< b->column;j++){
                    float temp = 0.0;
                    for (int k=0;k< a->column;k++){
                       temp += a->get(i,k) * b->get(k,j);
                    }
                    result.set(i,j,temp);
                }
            }
            return result;
        }
        //cache optimal matrix multiply
        static Matrix cache_multiply(Matrix *a, Matrix *b){
            a->modify_storing_style(store_style::row);
            b->modify_storing_style(store_style::column);   
            return uBLAS::naive_multiply(a,b);
        }
        //vectorized 
        static Matrix vectorization_multiply(Matrix *a, Matrix *b){
            if (a->column!=b->row){
                throw std::invalid_argument("Dimension mismatch");
            }
            a->modify_storing_style(store_style::row);
            b->modify_storing_style(store_style::column);

            Matrix result = Matrix(a->row, b->column);
            for (int i=0;i < a->row;i++){
                for (int j=0;j< b->column;j++){
                    __m256 temp = _mm256_setzero_ps();
                    //calculate till the right alignment
                    float temp2=0.0;
                    int startpos=0;
                    for (int k=0;((((i*a->column + k) % 8 != 0) || ((j*b->row + k) % 8 !=0)) && (k<a->column));k++){
                        startpos=k+1;
                        temp2 += a->get(i,k) * b->get(k,j);
                    }
                    int endpos=startpos;
                    for (int k=startpos;k + 8 <= a->column;k+=8){
                        __m256 v1= _mm256_load_ps(&(a->data[i*a->column + k]));
                        __m256 v2= _mm256_load_ps(&(b->data[j*b->row + k]));
                        temp = _mm256_fmadd_ps(v1, v2, temp);
                        endpos=k+8;
                    }
                    //calculate the last batch element
                    for (int k=0;k<8;k++){
                        temp2+=temp[k];
                    }
                    for (int k=endpos;k<a->column;k++){
                        temp2 += a->get(i,k) * b->get(k,j);
                    }
                    result.set(i,j,temp2);
                }
            }
            return result; 
        }
        
        static Matrix parallel_multiply(Matrix *a, Matrix *b, int numberOfProcessors, int rank){
            if (a->column!=b->row){
                throw std::invalid_argument("Dimension mismatch");
            }
            b->modify_storing_style(store_style::column);
            Matrix result = Matrix(a->row, b->column);
            //determine the optimal number of processors in vertical direction with some heuristics.
            //Heuristics: minimize the difference between the biggest and the smallest block + "shapedistance" from square blocks
            int best_row_nr=0;
            int best_value=2 * a->row *b->column;
            for(int nr_row=1;nr_row<=numberOfProcessors;nr_row++){
                //#calculate the bigges chunk size in the last row 
                int block_in_last_row = numberOfProcessors-(numberOfProcessors / nr_row)* (nr_row-1);
                int biggest_chunk = (a->row / nr_row) *(b->column / block_in_last_row);
                //difference of the avg row and col number in a block
                int ratio = pow((a->row / nr_row) - (b->column/(ceil(((float)numberOfProcessors)/nr_row))),2);  
                int loss = biggest_chunk + ratio;

                if (loss < best_value && block_in_last_row*nr_row<=numberOfProcessors){
                    best_value=loss;
                    best_row_nr = nr_row;
                } 
            }
            int best_column_nr = static_cast<int>(floor((float)numberOfProcessors / (float)best_row_nr));
            int local_size_A[numberOfProcessors]; 
            int local_size_B[numberOfProcessors];
            for (int r = 0;r<numberOfProcessors; r++){
                if ((r / best_column_nr) == best_row_nr-1){
                    //last row
                    local_size_A[r] = a->row - (best_row_nr-1)* (a->row /best_row_nr);
                    //last column
                    if (r == numberOfProcessors-1){
                        local_size_B[r] = b->column - (b->column / (numberOfProcessors-best_column_nr*(best_row_nr-1))) * (numberOfProcessors-best_column_nr*(best_row_nr-1)-1); 
                    }
                    else{
                        local_size_B[r] = b->column / (numberOfProcessors-best_column_nr*(best_row_nr-1));
                    }
                }
                else{
                    local_size_A[r] = (a->row /best_row_nr);
                    //last column
                    if ((r == numberOfProcessors-1) || ((r+1) % best_column_nr==0)){
                        local_size_B[r] = b->column - (b->column / best_column_nr) * (best_column_nr-1);
                    }
                    else{
                        local_size_B[r] = b->column / best_column_nr;
                    }
                }
            }
            
            Matrix A(local_size_A[rank], a->column);
            Matrix B(b->row, local_size_B[rank]);
            B.storing = store_style::column;
            //Scatter
            int countA[numberOfProcessors];
            int posA[numberOfProcessors];
            int countB[numberOfProcessors];
            int posB[numberOfProcessors];
            posA[0]=0;
            posB[0]=0;
            countA[0]=local_size_A[0] * a->column;
            countB[0]=local_size_B[0] * b->row;
            for (int r = 1;r<numberOfProcessors; r++){
                countA[r]=local_size_A[r] * a->column;
                countB[r]=local_size_B[r] * b->row;
                if (r % best_column_nr == 0)
                    posA[r]=posA[r-1]+countA[r-1];
                else
                    posA[r]=posA[r-1];
                if (r % best_column_nr == 0)
                    posB[r]=0;
                else
                    posB[r]=posB[r-1]+countB[r-1];
            }
            
            MPI_Scatterv(a->data, countA, posA, MPI_FLOAT,A.data, countA[rank], MPI_FLOAT,0, MPI_COMM_WORLD);
            MPI_Scatterv(b->data, countB, posB, MPI_FLOAT,B.data, countB[rank], MPI_FLOAT,0, MPI_COMM_WORLD);
            Matrix C = uBLAS::vectorization_multiply(&A,&B);
            //gather
            int countC[numberOfProcessors];
            int posC[numberOfProcessors];
            posC[0]=0;
            countC[0]=local_size_A[0] * local_size_B[0];
            for (int r = 1;r<numberOfProcessors; r++){
                countC[r]=local_size_A[r] *local_size_B[r];
                posC[r]=posC[r-1]+countC[r-1];
            }
            float temp[result.row*result.column];
            MPI_Gatherv(C.data, C.row * C.column, MPI_FLOAT,temp, countC, posC,MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (rank==0){
                int col_pos=0;
                for (int r=0;r<numberOfProcessors;r++){
                    if (r % best_column_nr ==0){
                        col_pos=0;
                    }
                    else{
                        col_pos+=local_size_B[r-1];
                    }
                    int block_start_pos =(r / best_column_nr) * local_size_A[0] *result.column + col_pos;
                    for (int row=0;row<local_size_A[r];row++){
                        memcpy ( &result.data[(block_start_pos + row * result.column)], &temp[posC[r]+row*local_size_B[r]], sizeof(float)*local_size_B[r]);
                    }
                }
            }
            return result;
        }
        

        // LU Matrix decomposition: Gauss elimination, return the P(permutation), L (lower tridiagonal) and U(upper tridiagonal) matrix
        static void naive_LU(Matrix *original,Matrix* L, Matrix* U){
            //work on U matrix
            U->clone_data(original);
            L->zero();
            for (int row=0;row<U->row;row++){
                //main diagonal
                L->set(row,row,1);
                for (int temp_row=row+1;temp_row<U->row;temp_row++){
                    float ratio= U->get(temp_row,row) / U->get(row,row);
                    L->set(temp_row,row,ratio);
                    U->set(temp_row,row,0.0);
                    for (int col=row+1;col<original->column;col++){
                        float new_value = U->get(temp_row, col) - ratio * U->get(row, col);
                        U->set(temp_row, col, new_value);
                    }
                }
            }
        }

        static void GAXPY_LU(Matrix *original, Matrix* L, Matrix* U){
            U->clone_data(original);
            L->zero();
            L->set(0,0,1);
            for (int temp_row=1;temp_row<U->row;temp_row++){
                L->set(temp_row,temp_row,1);
                for (int row=0;row<temp_row;row++){
                    //main diagonal
                           
                        float ratio= U->get(temp_row,row) / U->get(row,row);
                        L->set(temp_row,row,ratio);
                        U->set(temp_row,row,0.0);
                        for (int col=row+1;col<original->column;col++){
                            float new_value = U->get(temp_row, col) - ratio * U->get(row, col);
                            U->set(temp_row, col, new_value);
                        }
                    }
                
            }
        }

        //parallel LU decomposition
        static void parallel_LU(Matrix *original, Matrix* L, Matrix* U, int numberOfProcessors, int rank){
            U->clone_data(original);
            L->zero();
            int block_size=24;
            int from_corner = 0;
            for (int row=0;row+block_size<original->column;row=row+block_size){
                from_corner=row+block_size;
                Matrix block_original = U->get_subMatrix(row,block_size,row, block_size);
                Matrix U11(block_size, block_size);
                Matrix L11(block_size, block_size);
                uBLAS::GAXPY_LU(&block_original, &L11, &U11);
                L->set_subMatrix(row,row,&L11);
                U->set_subMatrix(row,row,&U11);
                Matrix U12(block_size, original->column-row-block_size);
                Matrix L21(original->row-row-block_size, block_size);
                //calc row and col
                if (rank==0){
                    for (int col_L=0;col_L<block_size;col_L++){
                        for (int row_L=0; row_L+row+block_size<L->row;row_L++){
                            float value=U->get(row+block_size+row_L,row+col_L);
                            for (int k=0;k<col_L;k++){
                                value-= L21.get(row_L,k) * U11.get(k,col_L);
                            }
                            L21.set(row_L,col_L,value  / U11.get(col_L,col_L));
                            //zero U in he right postion
                            U->set(row+block_size+row_L,row+col_L,0.0);
                        }
                    }

                    for (int row_U=0;row_U<block_size;row_U++){
                        for (int col_U=0; col_U+row+block_size<L->column;col_U++){
                            float value=U->get(row+row_U,row+block_size + col_U);
                            for (int k=0;k<row_U;k++){
                                value-= U12.get(k,col_U) * L11.get(row_U,k);
                            }
                            U12.set(row_U,col_U,value  / L11.get(row_U,row_U));
                        }
                    }

                
                    U->set_subMatrix(row,row+block_size,&U12);
                    L->set_subMatrix(row+block_size,row,&L21);
                }
                Matrix temp = uBLAS::parallel_multiply(&L21, &U12,numberOfProcessors, rank);
                //minus part
                for (int i=0;i<temp.row;i++){
                    for (int j=0;j<temp.column;j++){
                        U->set(row+block_size+i,row+block_size+j, U->get(row+block_size+i,row+block_size+j)-temp.get(i,j));
                    }
                }

            }
            Matrix corner = U->get_subMatrix(from_corner,original->row-from_corner, from_corner, original->column-from_corner);
            Matrix L_corner(original->row - from_corner, original->column - from_corner);
            Matrix U_corner(original->row - from_corner, original->column - from_corner);
            uBLAS::GAXPY_LU(&corner,&L_corner,&U_corner);
            L->set_subMatrix(from_corner,from_corner,&L_corner);
            U->set_subMatrix(from_corner,from_corner,&U_corner);
        }


        //QR decomposition: Gram-Smith orthogonalization
        static void QR(const Matrix a){

        }

};



