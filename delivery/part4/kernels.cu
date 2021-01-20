#ifndef KERNEL_REDUCE
#define KERNEL_REDUCE

#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat(float *h, float *g, float *residuals, int N) {

    float diff = 0.0;
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if (tidx > 0 && tidx < N - 1 && tidy > 0 && tidy < N - 1) {
        g[tidx * N + tidy] = 0.25 * (h[tidx * N + (tidy - 1)] +  // left
                                     h[tidx * N + (tidy + 1)] +  // right
                                     h[(tidx - 1) * N + tidy] +  // top
                                     h[(tidx + 1) * N + tidy]); // bottom
        diff = g[tidx * N + tidy] - h[tidx * N + tidy];
        residuals[tidx * N + tidy] = diff * diff;
    }
}


template<size_t blockSize, typename T>
__device__ void warpReduce(volatile T *sdata, size_t tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<int blockSize, class T>
__global__ void reduce1(T *g_idata, T *g_odata, int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;
    if (i < n) {
        sdata[tid] = g_idata[i];
    }

    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<int blockSize, class T>
__global__ void reduce2(T *g_idata, T *g_odata, int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = 0;
    if (i < n) {
        sdata[tid] = g_idata[i];
    }
	
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<int blockSize, class T>
__global__ void reduce3(T *g_idata, T *g_odata, int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;

    if (i < n) {
        sdata[tid] = g_idata[i];
    }

    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<int blockSize, class T>
__global__ void reduce4(T *g_idata, T *g_odata, int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = 0;

    if (i < n) {
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    }

    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<int blockSize, class T>
__global__ void reduce5(T *g_idata, T *g_odata, int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = 0;

    if (i < n) {
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    }
	
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<int blockSize, class T>
__global__ void reduce6(T *g_idata, T *g_odata, int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = 0;

    if (i < n) {
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    }
	
    __syncthreads();
    if (blockSize >= 512 && tid < 256) { sdata[tid] += sdata[tid + 256]; }
    __syncthreads();
    if (blockSize >= 256 && tid < 128) { sdata[tid] += sdata[tid + 128]; }
    __syncthreads();
    if (blockSize >= 128 && tid < 64) { sdata[tid] += sdata[tid + 64]; }
    __syncthreads();

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template<size_t blockSize, typename T>
__global__ void finalReduce(T *g_idata, T *g_odata, size_t n) {
    __shared__ T sdata[blockSize];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * (blockSize) + tid;
    size_t gridSize = blockSize * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += g_idata[i];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] += sdata[tid + 512]; }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template<size_t blockSize, typename T>
T GPUReductionOrg(T *dA, size_t N) {
    T tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1. * n) / blockSize);

    T *tmp;
    cudaMalloc(&tmp, sizeof(T) * blocksPerGrid);

    T *from = dA;

    do {
        blocksPerGrid = std::ceil((1. * n) / blockSize);
        finalReduce<blockSize><<<blocksPerGrid, blockSize, blockSize*sizeof(T)>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1)
        finalReduce<blockSize><<<1, blockSize, blockSize*sizeof(T)>>>(tmp, tmp, n);

    cudaDeviceSynchronize();

    cudaMemcpy(&tot, tmp, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(tmp);
    return tot;
}

#endif
