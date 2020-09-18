#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define DIM 512

/*
 * An example of using shared memory to optimize performance of a parallel
 * reduction by constructing partial results for a thread block in shared memory
 * before flushing to global memory.
 */

// Returns the current time in microseconds
long long start_timer() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec * 1000000 + tv.tv_usec;
}

// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char *name) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
        printf("%s: %.5f sec ", name, ((float) (end_time - start_time)) / (1000 * 1000));
        return end_time - start_time;
}

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

__global__ void reduceNeighboredGmem_1(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredSmem_2(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    smem[tid] = idata[tid];
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
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceNeighboredSmemNoDivergence_3(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            smem[index] += smem[index + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceInterleavedSmem_4(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceUnrolling2_5(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2 data blocks
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceUnrollingWarp8_6(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceCompleteUnrolling8_7(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnrolling8Template_8(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
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

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = DIM;   // initial block size

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );

    memcpy (tmp, h_idata, bytes);

    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    long long start_time = start_timer();
    int cpu_sum = recursiveReduce (tmp, size);
    stop_timer(start_time, "cpu_sum time");
    printf(" cpu_sum: %d\n", cpu_sum);

    // reduce reduceNeighboredGmem_1
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    start_time = start_timer();
    reduceNeighboredGmem_1<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    stop_timer(start_time, "reduceNeighboredGmem_1 time");
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf(" gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce reduceNeighboredSmem_2 
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    start_time = start_timer();
    reduceNeighboredSmem_2<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    stop_timer(start_time, "reduceNeighboredSmem_2 time");
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf(" gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce reduceNeighboredSmemNoDivergence_3 
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    start_time = start_timer();
    reduceNeighboredSmemNoDivergence_3<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    stop_timer(start_time, "reduceNeighboredSmemNoDivergence_3 time");
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf(" gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce reduceInterleavedSmem_4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    start_time = start_timer();
    reduceInterleavedSmem_4<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    stop_timer(start_time, "reduceInterleavedSmem_4 time");
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf(" gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce reduceUnrolling2_5
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    start_time = start_timer();
    reduceUnrolling2_5<<<grid.x/2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    stop_timer(start_time, "reduceUnrolling2_5 time");
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];
    printf(" gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x/2, block.x);

    // reduce reduceUnrollingWarp8_6
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    start_time = start_timer();
    reduceUnrollingWarp8_6<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    stop_timer(start_time, "reduceUnrollingWarp8_6 time");
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
    printf(" gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x/8, block.x);

    // reduce reduceCompleteUnrolling8_7
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    start_time = start_timer();
    reduceCompleteUnrolling8_7<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    stop_timer(start_time, "reduceCompleteUnrolling8_7 time");
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
    printf(" gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x/8, block.x);

    // reduce reduceCompleteUnrolling8Template_8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    start_time = start_timer();
    switch (blocksize) {
      case 1024:
        reduceCompleteUnrolling8Template_8<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
      case 512:
        reduceCompleteUnrolling8Template_8<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
      case 256:
        reduceCompleteUnrolling8Template_8<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
      case 128:
        reduceCompleteUnrolling8Template_8<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
      case 64:
        reduceCompleteUnrolling8Template_8<64><<<grid.x/8, block>>>(d_idata, d_odata, size);
        break;
    }
    cudaDeviceSynchronize();
    stop_timer(start_time, "reduceCompleteUnrolling8Template_8 time");
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
    printf(" gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x/8, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}

  