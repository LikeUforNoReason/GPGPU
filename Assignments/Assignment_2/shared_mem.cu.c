#include "./common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda.h>
#include "./book.h"

#define DIM 512

float gpu_res1 = 0.0;
float gpu_res2 = 0.0;

void random_floats(float *x, int size)
{ 
  for (int i = 0; i < size; i++) { 
    x[i] = (float)(rand()/(float)RAND_MAX);
  }
}

__global__ void reduceNeighboredSmem_2(float *g_adata, float *g_bdata, float *g_cdata, unsigned int  n)
{
    __shared__ float smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = 0.0;
    if (idx < n) {
    	smem[tid] = g_adata[idx]*g_bdata[idx];
    }
    __syncthreads();

    // in-place reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0){
            smem[tid] += smem[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_cdata[blockIdx.x] = smem[0];
}

__global__ void reduceNeighboredSmem_3(float *g_adata, float *g_bdata, float *g_cdata, unsigned int  n)
{
    __shared__ float smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = 0.0;
    if (idx < n) {
    	smem[tid] = g_adata[idx]*g_bdata[idx];
    }
    __syncthreads();

    // in-place reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0){
            smem[tid] += smem[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    //if (tid == 0) g_cdata[blockIdx.x] = smem[0];
    if (tid == 0) atomicAdd(g_cdata, smem[0]);
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = DIM;   // initial block size

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(float);
    float *h_adata = (float *) malloc(bytes);
    float *h_bdata = (float *) malloc(bytes);
    float *h_cdata = (float *) malloc(grid.x * sizeof(float));
    float *h_cdata1 = (float *) malloc(sizeof(float));

    srand ((unsigned) time (NULL));
  	random_floats(h_adata, size);
  	random_floats(h_bdata, size);

    // allocate device memory
    float *d_adata = NULL;
    float *d_bdata = NULL;
    float *d_cdata = NULL;
    float *d_cdata1 = NULL;
    CHECK(cudaMalloc((void **) &d_adata, bytes));
    CHECK(cudaMalloc((void **) &d_bdata, bytes));
    CHECK(cudaMalloc((void **) &d_cdata, grid.x * sizeof(float)));
    CHECK(cudaMalloc((void **) &d_cdata1, sizeof(float)));

    // reduce reduceNeighboredSmem_2 
    CHECK(cudaMemcpy(d_adata, h_adata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bdata, h_bdata, bytes, cudaMemcpyHostToDevice));
    //cudaMemcpy(tmp, &gpu_res2, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEvent_t    start1, stop1;
    HANDLE_ERROR( cudaEventCreate( &start1 ));
    HANDLE_ERROR( cudaEventCreate( &stop1 ));
    HANDLE_ERROR( cudaEventRecord( start1, 0 ));
    reduceNeighboredSmem_2<<<grid, block>>>(d_adata, d_bdata, d_cdata, size);
    cudaDeviceSynchronize();
    HANDLE_ERROR( cudaEventRecord( stop1, 0 ));
    HANDLE_ERROR( cudaEventSynchronize( stop1 ));
    float elapsedTime1;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime1, start1, stop1 ));
    printf("Kernel 1 generates time is:  %f ms\n", elapsedTime1);

    CHECK(cudaMemcpy(h_cdata, d_cdata, grid.x * sizeof(float), cudaMemcpyDeviceToHost));
    //float gpu_res1 = 0.0;
    for (int i = 0; i < grid.x; i++) gpu_res1 += h_cdata[i];
    printf(" gpu_res1: %f <<<grid %d block %d>>>\n", gpu_res1, grid.x, block.x);

	// reduce reduceNeighboredSmem_3 
    CHECK(cudaMemcpy(d_adata, h_adata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bdata, h_bdata, bytes, cudaMemcpyHostToDevice));
    //cudaMemcpy(d_cdata1, &gpu_res2, sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEvent_t    start2, stop2;
    HANDLE_ERROR( cudaEventCreate( &start2 ));
    HANDLE_ERROR( cudaEventCreate( &stop2 ));
    HANDLE_ERROR( cudaEventRecord( start2, 0 ));
    reduceNeighboredSmem_3<<<grid, block>>>(d_adata, d_bdata, d_cdata1, size);
    cudaDeviceSynchronize();
    //float tmp = 0.0;
    HANDLE_ERROR( cudaEventRecord( stop2, 0 ));
    HANDLE_ERROR( cudaEventSynchronize( stop2 ));
    float elapsedTime2;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime2, start2, stop2 ));
    printf("Kernel 2 generates time is:  %f ms\n", elapsedTime2);

    CHECK(cudaMemcpy(&gpu_res2, d_cdata1, sizeof(float), cudaMemcpyDeviceToHost));
    //float gpu_res1 = 0.0;
    //for (int i = 0; i < grid.x; i++) gpu_res1 += h_cdata[i];
    printf(" gpu_res2: %f <<<grid %d block %d>>>\n", gpu_res2, grid.x, block.x);

	//free events
	HANDLE_ERROR( cudaEventDestroy( start1 ));
	HANDLE_ERROR( cudaEventDestroy( stop1 ));
	HANDLE_ERROR( cudaEventDestroy( start2 ));
	HANDLE_ERROR( cudaEventDestroy( stop2 ));

    // free host memory
    free(h_adata);
    free(h_bdata);
    free(h_cdata);

    // free device memory
    CHECK(cudaFree(d_adata));
    CHECK(cudaFree(d_bdata));
    CHECK(cudaFree(d_cdata));
    CHECK(cudaFree(d_cdata1));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    //bResult = (gpu_res1 == gpu_res2);

    //if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}

  