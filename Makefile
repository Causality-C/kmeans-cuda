all: main

main: main.cpp kmeans.cu config.cpp config.h kmeans.h profile.cpp profile.h
	nvcc -o main main.cpp kmeans.cu config.cpp profile.cpp --extended-lambda -arch=sm_60

clean:
	rm -f main