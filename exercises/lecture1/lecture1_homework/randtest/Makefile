

NVCC=nvcc
NVOPTS=-arch sm_20

hello_world: kernel.o
	$(NVCC) $(NVOPTS) -o x.randtest kernel.o

kernel.o: kernel.cu
	$(NVCC) $(NVOPTS) -c kernel.cu 

clean:
	rm -rf kernel.o x.randtest
