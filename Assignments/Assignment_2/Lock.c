#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct Lock {
  int *mutex;
  Lock( void ) {
    int state = 0;
    cudaMalloc( (void**)& mutex, sizeof(int) );
    cudaMemcpy( mutex, &state, sizeof(int), cudaMemcpyHostToDevice );
  }

  ~Lock( void ) {
    cudaFree( mutex );
  }

  __device__ void lock( void ) {
    while( atomicCAS( mutex, 0, 1 ) != 0 );
  }

  __device__ void unlock( void ) {
   atomicExch( mutex, 0 );
  }
};

__global__ void blockCounterUnlocked(int *nblocks) {
  if (threadIdx.x == 0) {
    *nblocks = *nblocks + 1;
  }
}

__global__ void blockCounterLocked(Lock lock, int *nblocks) {
  if (threadIdx.x == 0) {
    lock.lock();
    *nblocks = *nblocks + 1;
    lock.unlock();
  }
}

int main(){
  int nblocks, *d_nblocks;
  Lock lock;

  cudaMalloc((void**) &d_nblocks, sizeof(int));

  // blockCounterUnlocked
  nblocks = 0;
  cudaMemcpy(d_nblocks, &nblocks, sizeof(int), cudaMemcpyHostToDevice);

  blockCounterUnlocked<<<512,1024>>>(d_nblocks);

  cudaMemcpy(&nblocks, d_nblocks, sizeof(int), cudaMemcpyDeviceToHost);

  printf("blockCountUnlocked counted %d blocks\n", nblocks);

  // blockCounterLocked
  nblocks = 0;
  cudaMemcpy(d_nblocks, &nblocks, sizeof(int), cudaMemcpyHostToDevice);

  blockCounterLocked<<<512,1024>>>(lock, d_nblocks);

  cudaMemcpy(&nblocks, d_nblocks, sizeof(int), cudaMemcpyDeviceToHost);

  printf("blockCountLocked counted %d blocks\n", nblocks);

  cudaFree(d_nblocks);
}
