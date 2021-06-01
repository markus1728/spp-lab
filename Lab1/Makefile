CC=gcc
CXX=g++
CFLAGS=-O2 -fopenmp -std=c99
CXXFLAGS=-O2 -fopenmp
EXECS= hello-openmp dotproduct quickSortSequential quickSortParallel heated-plate-parallel

all: $(EXECS)

hello-openmp: hello-openmp.c
	$(CC) -o $@ $<  $(CFLAGS)

dotproduct: dotproduct.c  
	$(CC) -o $@ $< $(CFLAGS) -lm

quickSortSequential: quickSortSequential.c  
	$(CC) -o $@ $< $(CFLAGS)

quickSortParallel: quickSortParallel.c  
	$(CC) -o $@ $< $(CFLAGS)

heated-plate-parallel: heated-plate-parallel.cpp  
	$(CXX) -o $@ $< $(CXXFLAGS)

clean:
	rm -f $(EXECS) *.o
