install:
	sudo apt install mpich

compile:
	#mpic++ -mavx2 -mfma -O3 main.cpp -o main
	nvcc  main.cu -O3 -o main_gpu


test:
	nvcc  test.cu -O3 -o test_gpu
	./test_gpu
	mpic++ -mavx2 -mfma -O3 test.cpp -o test
	mpirun -np 2 ./test

clean:
	rm -rf test
	rm -rf main
