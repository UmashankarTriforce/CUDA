libelementwise.so: elementwise.o
	nvcc -Xcompiler -fPIC --shared -o libelementwise.so elementwise.o -lcudart
	rm -f *.o

elementwise.o: elementwise.cu
	nvcc -Xcompiler -fPIC -x cu -rdc=true -c elementwise.cu -o elementwise.o

clean :
	rm -f *.o *.so