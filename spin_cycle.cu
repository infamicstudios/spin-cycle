#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <unistd.h>

#define HISTOGRAM_SIZE 32  // clock values are 32 bit
#define THREADS_PER_BLOCK 256
#define BLOCKS_NUM 32

__device__ unsigned int histogram[HISTOGRAM_SIZE];

__global__ void spin(unsigned long int reps) {
  __shared__ unsigned int sharedHistogram[HISTOGRAM_SIZE];

  // Init shared memory
  for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
    sharedHistogram[i] = 0;
  }
  __syncthreads();

  for (unsigned int i = 0; i < reps; i++) {
    __syncthreads();
    unsigned int t1, t2;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t1)::"memory");
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t2)::"memory");

    //Log_2 binning
    atomicAdd(&sharedHistogram[__clz(t2 - t1)], 1);
  }

  __syncthreads();

  // move shared into global mem
  for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
    atomicAdd(&histogram[i], sharedHistogram[i]);
  }
}

// Runner
void runHistogram(unsigned long int reps) {
  unsigned int h_histogram[HISTOGRAM_SIZE] = {0};

  int nDevices;
  cudaGetDeviceCount(&nDevices);

  for (int i = 0; i < nDevices; i++) {
    cudaSetDevice(i);  // change active gpu

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    // Header
    // Ripped from https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
    printf("Device Number: %d\n", i);
    printf("\tName: %s\n", prop.name);
    printf("\tMemory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("\tPeak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("\t# Blocks (hardcoded) %d\n", BLOCKS_NUM);
    printf("\t# Threads per block (hardcoded) %d\t", THREADS_PER_BLOCK);
    printf("\t# repetitions %d\t", reps);

    // Reset (Not sure if this is needed)
    cudaMemset(histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int));

    // Launch kernel
    spin<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(reps);

    // Copy results back to host
    cudaMemcpy(h_histogram, histogram, HISTOGRAM_SIZE * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    
    // Print Results
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
      printf("Bin %d: %u\n", i, h_histogram[i]);
    }
  }
}


int main(int argc, char* argv) {
  while ((opt = getopt(argc, argv, "n:")) != -1) {
    switch (opt) {
      case 'n':
        reps = atoi(optarg);  
        break;
      default:
        fprintf(stderr, "Usage: %s [-n integer number of repetitions]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
  runHistogram(reps);
  return 1;
}
