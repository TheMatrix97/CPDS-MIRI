#include <math.h>
#include <float.h>
#include <cuda.h>
#include <stdio.h>


__global__ void gpu_Heat (float *h, float *g, int N) {
	//Calculate index for this thread
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	//Compute heat equation jacobi
	g[i*N+j]= 0.25f * (h[i*N + (j-1)]+  // left
					     h[i*N  + (j+1)]+  // right
				             h[(i-1)*N + j]+  // top
				             h[(i+1)*N + j]); // bottom
	
}


__global__ void reduce0 (float *u, float *uhelp, float *res, int N){
	extern __shared__ float sdata[];
	
	//each thread loads one element from global to shared mem
	int blockId = blockIdx.x * gridDim.x + blockIdx.y;
	int tid = threadIdx.x * blockDim.x + threadIdx.y;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	float diff = uhelp[i*N+j] - u[i*N+j];
	sdata[tid] = diff*diff;
	//printf("Setting TID %i // i=%i ; j=%i -> %f // \n", tid, i, j, diff*diff); //some diff is != 0
	__syncthreads(); //at this point all threads have loaded the element
	
	//do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x*blockDim.y; s *= 2){
		if (tid % (2*s) == 0){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	//write result for this block to global mem
	if (tid == 0){ //Always return sdata[0] = 0 why??
		//printf("res --> id: %i --> %f \n", blockId ,sdata[0]);
		res[blockId] = sdata[0];
	}
}
