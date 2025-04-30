all: main

main: main.cpp kernel.cu
	nvcc -o main main.cpp kernel.cu --extended-lambda -arch=sm_60

clean:
	rm -f main