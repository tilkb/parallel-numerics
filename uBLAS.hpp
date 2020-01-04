#include <stdexcept>
#include <math.h>
#include <immintrin.h>
#include "mpi.h"
#include "matrix.h"
#include <assert.h>

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
                    float dot_product = 0.0;
                    for (int k=0;k< a->column;k++){
                       dot_product += a->get(i,k) * b->get(k,j);
                    }
                    result.set(i,j,dot_product);
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
            float copy_area[result.row*result.column];
            MPI_Gatherv(C.data, C.row * C.column, MPI_FLOAT,copy_area, countC, posC,MPI_FLOAT, 0, MPI_COMM_WORLD);
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
                        std::memcpy ( &result.data[(block_start_pos + row * result.column)], &copy_area[posC[r]+row*local_size_B[r]], sizeof(float)*local_size_B[r]);
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


        //QR decomposition: Gram-Schmidt orthogonalization
        static void QR(Matrix* a,Matrix* Q, Matrix* R){
            R->zero();
            Q->clone_data(a);
            for (int i=0;i<a->column;i++){
                for (int j=0;j<i;j++){
                    //calculate the projections
                    float nominator=0;
                    float denominator=0;
                    for(int k=0;k<a->row;k++){
                       nominator+= Q->get(k,i) * Q->get(k,j);
                       denominator+=Q->get(k,j) * Q->get(k,j);
                    }
                    float ratio = nominator / denominator;
                    //decrease the vector
                    for(int k=0;k<a->row;k++){
                        Q->set(k,i,Q->get(k,i) - ratio * Q->get(k,j));
                    }
                    R->set(j,i,ratio);

                }
                //make vector length to 1
                float vec_len=0;
                for(int k=0;k<Q->row;k++){
                    vec_len+=Q->get(k,i)*Q->get(k,i);
                }
                vec_len=sqrt(vec_len);
                for(int k=0;k<Q->row;k++){
                    Q->set(k,i,Q->get(k,i)/vec_len);
                }
                R->set(i,i,vec_len);
            }
        }
        //QR decomposition: Hausholder orthogonalization
        static void QR_hausholder(Matrix* a,Matrix* Q, Matrix* R, int numberOfProcessors, int rank){
            R->clone_data(a);
            //make it identical
            Matrix I(Q->row,Q->column);
            Q->zero();
            I.zero();
            for (int i=0;i<Q->column;i++){
                I.set(i,i,1.0);
                Q->set(i,i,1.0);
            }
            for (int i=0;i<a->column;i++){
                //calculate the correct Hausholder matrix
                Matrix v(a->row,1);
                for (int k=0;k<a->row;k++){
                    v.data[k]=a->get(k,i);
                }
                // -alpha*e part
                v.data[i]= v.data[i] - v.get_vector_norm();

                v.normalize_vector();
                Matrix v_t = v;
                v_t.transpose();
                Matrix outer_product = uBLAS::vectorization_multiply(&v,&v_t);
                outer_product.const_multiply(2.0);
                Matrix H = I- outer_product;
                Matrix R_temp = uBLAS::parallel_multiply(&H,R, numberOfProcessors, rank);
                R->clone_data(&R_temp);

                Matrix Q_temp = uBLAS::parallel_multiply(Q,&H, numberOfProcessors, rank);
                Q->clone_data(&Q_temp);
            }
        }

        static Matrix richardson_iteration(Matrix* A, Matrix* b){
            float eps=0.0000001;
            Matrix x(A->row,1);
            x.zero();
            //calculate w precondidioner for A matrix to converge --> ||I-wA||<1
            float w = 1.0 / (0.1+ A->get_vector_norm());

            int iteration=0; 
            while(iteration<10000){
                Matrix gaxpy=uBLAS::vectorization_multiply(A,&x);
                Matrix add_part = (*b)-gaxpy;
                add_part.const_multiply(w);
                Matrix x_temp = x+add_part;
                x.clone_data(&x_temp);
                
                Matrix add_part_T =add_part;
                add_part_T.transpose();
                float error=uBLAS::vectorization_multiply(&add_part_T,&add_part).get(0,0)*(1.0/w);
                if (error<eps){
                    return x;
                }
                iteration++;
            }
            return x;
        }

        static Matrix jacobi_iteration(Matrix* A, Matrix* b){
            float eps=0.00000001;
            //optimize for sparse diagonal inverse
            float D_1[A->row];
            for (int i = 0;i < A->row;i++){
                D_1[i] = 1.0 / A->get(i,i);
            }
            Matrix x(A->row,1);
            x.zero();

            Matrix LU = (*A);
            for (int i = 0;i < A->row;i++){
                LU.set(i,i,0.0) ;
            }

            int iteration=0;
            bool need_iter=true;
            while(iteration<10000 && need_iter){
                need_iter=false;
                Matrix temp = uBLAS::vectorization_multiply(&LU, &x);
                for (int i=0;i<b->row;i++){
                    float x_new = (b->get(i,0)-temp.get(i,0))*D_1[i];
                    if (abs(x.get(i,0)-x_new)>eps){
                        need_iter=true;
                    }
                    x.set(i,0,x_new);
                }
                iteration++;
            }
            return x;
            
        }

        static Matrix gradient_method(Matrix* A, Matrix* b){
            float eps=0.0003;
            Matrix x(A->row,1);
            x.zero();
            int iteration=0;
            while(iteration<2000){
                Matrix d = (*b)-uBLAS::vectorization_multiply(A, &x);
                Matrix d_t =d;
                d_t.transpose();
                //calculate stepsize
                Matrix temp = uBLAS::vectorization_multiply(A, &d);
                float nominator = uBLAS::vectorization_multiply(&d_t,&d).get(0,0);
                float alpha = nominator / uBLAS::vectorization_multiply(&d_t, &temp).get(0,0);
                bool small_err=true;
                //check convergence
                for(int i=0;i<d.row;i++){
                    if (abs(d.data[i])>eps)
                        small_err=false;
                }
                if (small_err){
                    return x;
                }
                for (int i=0;i<d.row;i++){
                    d.data[i]*=alpha;
                    x.data[i]+=d.data[i];
                }
                iteration++;

            }
            return x;
        }

        static Matrix conjugate_gradient_method(Matrix* A, Matrix* b){
            float eps=0.00001;
            Matrix x(A->row,1);
            x.zero();
            Matrix p =(*b)-uBLAS::vectorization_multiply(A, &x);
            Matrix r = p;
            int iteration=0;
            while(iteration<1000){
                Matrix p_t = p;
                p_t.transpose();
                Matrix r_t = r;
                r_t.transpose();
                float nominator = uBLAS::vectorization_multiply(&r_t,&r).get(0,0);
                Matrix temp = uBLAS::vectorization_multiply(A, &p);
                float alpha = - nominator / uBLAS::vectorization_multiply(&p_t, &temp).get(0,0);
                for (int i=0;i<x.row;i++){
                    x.data[i]-=p.data[i] *alpha;
                    r.data[i]+=alpha * temp.data[i];
                }
                //calculate inner product of r,rT
                float new_nominator = 0;
                for (int i=0;i<r.row;i++)
                    new_nominator+=r.data[i]*r.data[i];

                if (new_nominator<eps){
                    return x;
                }
                float beta = new_nominator /nominator;
                for (int i=0;i<p.row;i++){
                    p.data[i]=beta*p.data[i] + r.data[i];
                }
                iteration++;

            }
            return x;
        }
        
    static float max_eig(Matrix* A, Matrix* eig_vector){
        float eps=0.0001;
        if (A->column!=A->row){
                throw std::invalid_argument("eigenvalues search works only for square matrix");
        }
        eig_vector->init_random();
        int iteration=0;
        bool close = false;
        float eig_val =0.0;
        while (iteration<10000 && close==false){
            eig_vector->normalize_vector();
            Matrix out = uBLAS::vectorization_multiply(A,eig_vector);
            //check distance
            eig_val = out.get(0,0)/eig_vector->get(0,0);
            eig_vector->const_multiply(eig_val);
            if (eig_vector->equals(&out,eps)){
                close=true;
            }
            eig_vector->clone_data(&out);

            iteration++;
        }
        return eig_val;
    }

    //need symetric matrix--> return eig values
    //in case of multi process mode: #of cols/rows must be n*(#processor*2)
    static Matrix eig_jacobi(Matrix* A, int numberOfProcessors, int rank){
        float eps=0.01;
        if (A->column!=A->row){
            throw std::invalid_argument("eigenvalues search works only for square matrix");
        }

        Matrix m = *A;
        //send the data to every: proc->PROC0 is the main node
        if (numberOfProcessors>0)
            MPI_Bcast(m.data,A->row*A->column,MPI_FLOAT,0,MPI_COMM_WORLD);

        //determine the current frobeneus norm of the A matrix
        float original_norm= m.get_vector_norm();

        int iteration = 0;
        bool done =false;
        while (iteration < 5000 && !done){
            //single core
            if (numberOfProcessors==1){
                for (int x=0;x<m.row-1;x++){
                    for (int y=x+1;y<m.column;y++){
                        float t = uBLAS::givens_rotation_left(&m,x,y);
                        uBLAS::givens_rotation_right(&m,x,y,t);
                    }
                }
            }
            //multi processor 
            else{
                assert(A->row %(numberOfProcessors*2)==0);
                //set pairing list
                int pairing[A->row];
                for (int i=0;i<A->row;i++)
                    pairing[i]=i;
                
                for (int inner_iter =0;inner_iter<A->row-1;inner_iter++){
                    int pair_from=A->row/numberOfProcessors*rank;
                    int pair_to=A->row/numberOfProcessors*(rank+1)-1;    

                    float send_collected_modification[A->row*(pair_to-pair_from+1)];
                    float t[(pair_to-pair_from+1)/2];
                    for (int i=0;i<(pair_to-pair_from+1)/2;i+=1){
                        t[i] = uBLAS::givens_rotation_left(&m,pairing[pair_from+i*2],pairing[pair_from+i*2+1]);
                        //collect data for sending
                        for (int j=0;j<A->column;j++){
                            send_collected_modification[i*2*A->row+j]=m.get(pairing[pair_from+i*2],j);
                            send_collected_modification[(i*2+1)*A->row+j]=m.get(pairing[pair_from+i*2+1],j);
                        }
                    }
                    uBLAS::data_broadcast_eig(&m, send_collected_modification, pairing, pair_from, pair_to, store_style::row, numberOfProcessors, rank);
                    for (int i=0;i<(pair_to-pair_from+1)/2;i++){
                        uBLAS::givens_rotation_right(&m,pairing[pair_from+i*2],pairing[pair_from+i*2+1],t[i]);
                        //collect data for sending
                        for (int j=0;j<A->column;j++){
                            send_collected_modification[i*2*A->row+j]=m.get(j,pairing[pair_from+i*2]);
                            send_collected_modification[(i*2+1)*A->row+j]=m.get(j,pairing[pair_from+i*2+1]);
                        }
                    }
                    uBLAS::data_broadcast_eig(&m, send_collected_modification, pairing, pair_from, pair_to, store_style::column, numberOfProcessors, rank);                 
                    //step pairing
                    //lower row
                    int left_lower=pairing[1];
                    for(int i=1;i<A->row-2;i+=2){
                        pairing[i]=pairing[i+2];
                    }
                    pairing[A->row-1]=pairing[A->row-2];
                    //upper row
                    for(int i=A->row-2;i>2;i-=2){
                        pairing[i] = pairing[i-2];
                    }
                    pairing[2]=left_lower;
                }
            }
            if (original_norm-m.get_diagonal_norm()<eps) 
                done=true;
            iteration++;
        }
        Matrix eig_values(A->row,1);
        for (int i=0;i<A->row;i++){
            eig_values.set(i,0,m.get(i,i));
        }
        return eig_values;
    }

    static Matrix eig_vectors_from_eig_vals(Matrix* A, Matrix* eig_values, int numberOfProcessors, int rank){
        if (numberOfProcessors>1)
            MPI_Bcast(A->data,A->row*A->column,MPI_FLOAT,0,MPI_COMM_WORLD);

        Matrix eig_vectors(A->row,A->row);
        if (A->row==1){
            eig_vectors.set(0,0,1.0);
            return eig_vectors;
        }      
        Matrix zero_vec(A->row,1);
        zero_vec.zero();
        
        //single process
        if (numberOfProcessors==1){ 
            for (int i=0;i<A->row;i++){
                //calculate eig vectors
                Matrix eig_A = *A;
                for (int j=0;j<A->row;j++){
                    eig_A.set(j,j,eig_A.get(j,j)- eig_values->get(i,0));
                }
                //modify a to avoid trivial solution: push away from zero solution
                zero_vec.set(0,0,0.004);
                Matrix eig_vec = uBLAS::conjugate_gradient_method(&eig_A,&zero_vec);
                //copy vector to the matrix
                for (int j=0;j<eig_vec.row;j++){
                    eig_vectors.set(j,i,eig_vec.get(j,0));
                }
            }
        }
        else{
            int count[numberOfProcessors];
            int pos[numberOfProcessors];
            for (int i=0;i<numberOfProcessors-1;i++){
                count[i]=(A->row/numberOfProcessors);
                pos[i]=count[i]*i;
            }
            count[numberOfProcessors-1]=A->row - (A->row/numberOfProcessors)*(numberOfProcessors-1);
            pos[numberOfProcessors-1]=A->row-count[numberOfProcessors-1];

            //scatter eig_values
            float eig_vals[count[rank]];

            MPI_Scatterv(eig_values->data, count, pos, MPI_FLOAT,eig_vals, count[rank], MPI_FLOAT,0, MPI_COMM_WORLD);
            Matrix out(A->row,count[rank]);
            out.storing=store_style::column;
            eig_vectors.storing=store_style::column;
            for (int i=0;i<count[rank];i++){
                //calculate eig vectors
                Matrix eig_A = *A;
                for (int j=0;j<A->row;j++){
                    eig_A.set(j,j,eig_A.get(j,j)- eig_vals[i]);
                }
                //modify a to avoid trivial solution: push away from zero solution
                zero_vec.set(0,0,0.004);
                Matrix eig_vec = uBLAS::conjugate_gradient_method(&eig_A,&zero_vec);
                //copy vector to the matrix
                for (int j=0;j<eig_vec.row;j++){
                    out.set(j,i,eig_vec.get(j,0));
                }
            }
            for (int i=0;i<numberOfProcessors;i++){
                count[i]*=A->row;
                pos[i]*=A->row;
            }
            
            MPI_Gatherv(out.data, count[rank], MPI_FLOAT,eig_vectors.data, count, pos, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }


        return eig_vectors;
    }

private:
    static void data_broadcast_eig(Matrix* A, float* send_collected_modification,int* pairing,int pair_from, int pair_to, store_style store_alignment, int numberOfProcessors, int rank){
        float recv_collected_modification[A->row*A->row];
        //communication parameters
        int amount_of_data_each_process[numberOfProcessors];
        int start_pos [numberOfProcessors];
        for (int i=0;i<numberOfProcessors;i++){
            amount_of_data_each_process[i]=A->row*A->row/numberOfProcessors;
            start_pos[i]=amount_of_data_each_process[i]*i;
        }
        MPI_Gatherv(send_collected_modification, A->row*(pair_to-pair_from+1), MPI_FLOAT,recv_collected_modification, amount_of_data_each_process, start_pos,MPI_FLOAT, 0, MPI_COMM_WORLD);
        //recover the matrix
        if (rank==0){
            for (int i=0;i<A->row;i++){
                for (int j=0;j<A->column;j++){
                    if (store_alignment==store_style::row)
                        A->set(pairing[i],j,recv_collected_modification[i*A->row+j]);
                    else
                        A->set(j,pairing[i],recv_collected_modification[i*A->row+j]);
                }
            }
        }
        MPI_Bcast(A->data,A->row*A->column,MPI_FLOAT,0,MPI_COMM_WORLD);

    }

    //Do left Givens rotation in place -> G_T * A
    static float givens_rotation_left(Matrix* a, int i, int j){
        if (i>=a->row || j>=a->row || a->row!=a->column || i<0 || j<0){
            throw std::invalid_argument("wrong parameters for Givens rotation");
        }
        float teta = (a->get(j,j) - a->get(i,i)) / (2*a->get(i,j));
        float t =  1.0/(abs(teta) + sqrt(1+teta*teta));
        if (teta<0)
            t=-t;
        float cos_teta = 1.0/(sqrt(t*t+1));
        float sin_teta = cos_teta*t;
        Matrix temp(a->row,2);
        for (int k=0;k<a->column;k++){
            temp.set(k,0,cos_teta*a->get(i,k) - sin_teta*a->get(j,k));
            temp.set(k,1,cos_teta*a->get(j,k) + sin_teta*a->get(i,k));
        }
        for (int k=0;k<a->column;k++){
            a->set(i,k,temp.get(k,0));
            a->set(j,k,temp.get(k,1));
        }
        return t;
    }

    //Do right Givens rotation in place -> A * G
    static void givens_rotation_right(Matrix* a, int i, int j, float t){
        if (i>=a->row || j>=a->row || a->row!=a->column || i<0 || j<0){
            throw std::invalid_argument("wrong parameters for Givens rotation");
        }
        float cos_teta = 1.0/(sqrt(t*t+1));
        float sin_teta = cos_teta*t;
        Matrix temp(a->row,2);
        for (int k=0;k<a->column;k++){
            temp.set(k,0,cos_teta*a->get(k,i) - sin_teta*a->get(k,j));
            temp.set(k,1,cos_teta*a->get(k,j) + sin_teta*a->get(k,i));
        }
        for (int k=0;k<a->column;k++){
            a->set(k,i,temp.get(k,0));
            a->set(k,j,temp.get(k,1));
        }

    }

};



