#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	g[i*N+j]= 0.25 * (h[i*N + (j-1)]+  // left
					     h[i*N  + (j+1)]+  // right
				             h[(i-1)*N + j]+  // top
				             h[(i+1)*N + j]); // bottom
	
}
