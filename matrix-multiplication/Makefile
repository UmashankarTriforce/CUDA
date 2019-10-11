libdgemm.so: dgemm.o
	nvcc -Xcompiler -fPIC --shared -o libdgemm.so dgemm.o -lcudart -lcublas
	rm -f *.o

dgemm.o: matrixMultiplication.cu
	nvcc -Xcompiler -fPIC -x cu -rdc=true -c matrixMultiplication.cu -o dgemm.o

clean :
	rm -f *.o *.so