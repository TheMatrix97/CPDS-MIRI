NVCC 	= /usr/local/cuda-11.1/bin/nvcc
CUDAPATH = /usr/local/cuda-11.1

NVCCFLAGS = -I$(CUDAPATH)/include #-Wdeprecated-declarations #--ptxas-options=-v
LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

ALL	= heatCUDA
all: clean $(ALL) run

kernels.o: kernels.cu
	$(NVCC) -c -g $(NVCCFLAGS) $+ $(LFLAGS) -o $@

heatCUDA: heatCUDA.cu #kernels.o
	$(NVCC) -g $(NVCCFLAGS) $+ $(LFLAGS) -o $@

run: ./heatCUDA $(ARGS)

clean:
	rm -fr $(ALL) *.o *.prv *.pcf *.row *.sym *.mpits set-0

