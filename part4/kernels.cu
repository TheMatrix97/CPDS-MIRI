#include <math.h>
#include <float.h>
#include <cuda.h>
#include <stdio.h>


__global__ void gpu_Heat (float *h, float *g, int N) {
	//Calculate index for this thread
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	//Compute heat equation jacobi
	g[i*N+j]= 0.25 * (h[i*N + (j-1)]+  // left
					     h[i*N  + (j+1)]+  // right
				             h[(i-1)*N + j]+  // top
				             h[(i+1)*N + j]); // bottom
	
}


__global__ void reduce0 (float *u, float *uhelp, float *res, int N){
	extern __shared__ float sdata[];
	
	//each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	float diff = uhelp[i*N+j] - u[i*N+j];
	sdata[tid] = diff*diff;
	printf("Setting TID %i // i=%i ; j=%i -> %f // \n", tid, i, j, diff*diff);
	__syncthreads(); //at this point all threads have loaded the element
	
	//do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x*blockDim.y; s *= 2){
		if (tid % (2*s) == 0){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	//write result for this block to global mem
	if (tid == 0){
		res[blockIdx.x] = sdata[0];
	}
}
