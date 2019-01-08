#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>

enum store_style {row, column};
//Sequential implementation of matrix operations.
class Matrix{
    public:
        int row;
        int column;
        store_style storing;
        float* data; //public because of the memory operands
        Matrix(int row, int column){
            this->row = row;
            this->column = column;
            //provide 32 bit alignment
            this->data = static_cast<float*>(_mm_malloc(sizeof(float) * row*column , 32));  //direct access can make it faster... need for Intel vector operations
            this->storing = store_style::row;
        };

        Matrix(const Matrix &m){
            this->row = m.row;
            this->column = m.column;
            this->storing = m.storing;
            this->data = static_cast<float*>(_mm_malloc(sizeof(float) * m.row*m.column , 32));  
            memcpy ( this->data, m.data, sizeof(float)*m.row*m.column);
            this->storing = store_style::row;
        }

        void modify_storing_style(store_style store){
            if (this->storing == store_style::row  && store == store_style::column){
                float* new_array = static_cast<float*>(_mm_malloc(sizeof(float) * this->row*this->column , 32));
                this->storing = store_style::column;
                for(int i=0;i<this->row;i++){
                    for (int j=0;j<this->column;j++){
                        new_array[j*this->row + i] = this->data[i*this->column + j];
                    }
                }
                delete(this->data);
                this->data = new_array;
            }
            else if(this->storing == store_style::column  && store == store_style::row){
                float* new_array = static_cast<float*>(_mm_malloc(sizeof(float) * this->row*this->column , 32));
                this->storing = store_style::row;
                for(int i=0;i<this->row;i++){
                    for (int j=0;j<this->column;j++){
                        new_array[i*this->column + j] = this->data[j*this->row + i];
                    }
                }
                delete(this->data);
                this->data = new_array;
            }

        }

        void print() {
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    std::cout << this->get(i,j) << "; ";
                }
                std::cout << std::endl;
            }
        }

        void init_random(){
            std::mt19937 rng;
            rng.seed(std::random_device()());
            std::uniform_real_distribution<> generator(-20,20);
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    float rand = generator(rng);
                    this->set(i,j,rand);
                }
            }
        }
        void init_SPD(){
            std::mt19937 rng;
            rng.seed(std::random_device()());
            std::uniform_real_distribution<> generator(0.001,20);
            for (int i=0;i<this->row;i++){
                for (int j=0;j<=i;j++){
                    float rand = generator(rng);
                    this->set(i,j,rand);
                    this->set(j,i,rand);
                }
            }
        }
        
        void transpose(){
            if (this->storing == store_style::row){
                this->storing = store_style::column;
            }
            else{
               this->storing = store_style::row; 
            }
            int temp = this->row;
            this->row = this->column;
            this->column = temp;
        }

        bool equals(Matrix* b){
            if ((this->row != b->row) || (this->column != b->column)){
                return false;
            }
            float eps = 0.5;
            for (int i=0;i<this->row;i++){
                for(int j=0;j<this->column;j++){
                    if (abs(this->get(i,j)-b->get(i,j))>eps){
                        return false;
                    } 
                }
            }
            return true;
        }

        void zero(){
            for (int i=0;i<this->row*this->column;i++){
                this->data[i] = 0.0;
                
            }
        }
        void clone_data(Matrix* from){
            memcpy(this->data,from->data,sizeof(float)*this->row*this->column);
            this->storing=from->storing;
        }
        void set_subMatrix(int row_from, int column_from, Matrix* submatrix){
            for (int i=0; i<submatrix->row;i++){
                for (int j=0; j<submatrix->column;j++){
                    this->set(row_from+i,column_from+j,submatrix->get(i,j));
                }

            }
        }

        Matrix get_subMatrix(int row_from,int row_nr,int column_from, int column_nr){
            if ((row_from<0) || (row_from+row_nr-1>=(this->row)) || (column_from<0) || (column_from+column_nr-1>=(this->column))){
                throw std::invalid_argument( "Out of range");
            }
            Matrix result(row_nr, column_nr);
            for (int i=0; i<row_nr;i++){
                for (int j=0; j<column_nr;j++){
                    result.set(i,j,this->get(row_from+i,column_from+j));
                }

            }
            return result;
        }


        float get(const int x,const int y){
            if ((x<0) || (x>=(this->row)) || (y<0) || (y>=(this->column))){
                throw std::invalid_argument( "Out of range" );
            }
            if (this->storing == store_style::row){
                return this->data[x*this->column + y];
            }
            else{
                return this->data[x + y *this->row];
            }
        } 
        
        void set(const int x,const int y, float value){
            if ((x<0) || (x>=(this->row)) || (y<0) || (y>=(this->column))){
                throw std::invalid_argument( "Out of range" );
            }
            if (this->storing == store_style::row)
                this->data[x*this->column + y] = value;
            else{
                this->data[x + y *this->row] = value;
            }
            
        }

        Matrix operator+(Matrix m2){
            if ((this->row!=m2.row) && (this->column!=m2.column)){
                throw std::invalid_argument("Size doesn't match");
            }
            Matrix result(this->row,this->column);
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    result.set(i,j,this->get(i,j)+m2.get(i,j));
                }
            }
            return result;
        }

        Matrix operator-(Matrix m2){
            if ((this->row!=m2.row) && (this->column!=m2.column)){
                throw std::invalid_argument("Size doesn't match");
            }
            Matrix result(this->row,this->column);
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    result.set(i,j,this->get(i,j)-m2.get(i,j));
                }
            }
            return result;
        }
        float get_vector_norm(){
            float vec_len=0;
            for(int k=0;k<this->row*this->column;k++){
                vec_len+=this->data[k] * this->data[k];
            }
            vec_len=sqrt(vec_len);
            return vec_len;
        }

        ///make vector norm 1 in place
        void normalize_vector(){
            float vec_len= this->get_vector_norm();
            for(int k=0;k<this->row*this->column;k++){
                this->data[k]=this->data[k] / vec_len;
            }
        }

        //in place: A=alpha*A
        void const_multiply(const float alpha){
            for(int k=0;k<this->row*this->column;k++){
                this->data[k]*=alpha;
            }
        }

        ~Matrix(){
            delete(this->data);
        }

};


