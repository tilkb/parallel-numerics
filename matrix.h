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
            this->data = static_cast<float*>(_mm_malloc(sizeof(float) * row*column , 32));
            this->storing = store_style::row;
        };

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
            std::uniform_real_distribution<> generator(-100,100);
            for (int i=0;i<this->row;i++){
                for (int j=0;j<this->column;j++){
                    float rand = generator(rng);
                    this->set(i,j,rand);
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
            for (int i=0;i<from->row;i++){
                for (int j=0;j<from->column;j++){
                    this->set(i,j,from->get(i,j));
                }
            }
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

        ~Matrix(){
            delete(this->data);
        }

};


