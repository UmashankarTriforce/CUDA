cudaPATH=/opt/cuda

all: sort

sort: cudaSort.o
	g++ -o sort -I$(cudaPATH)/include -L$(cudaPATH)/lib64 -lcuda -lcudart sort.cpp cudaSort.o

cudaSort.o:
	nvcc -c my_sort.cu -o cudaSort.o

clean: 
	rm -f *.o sort
