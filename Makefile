compile:
	mpic++ -mavx2 -mfma -O3 main.cpp -o main

test:
	mpic++ -mavx2 -mfma -O3 test.cpp -o test
	mpirun -np 2 ./test

clean:
	rm -rf test
	rm -rf main
