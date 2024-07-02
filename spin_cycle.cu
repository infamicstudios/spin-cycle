#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#define HISTOGRAM_SIZE 32  // clock values are 32 bit
#define THREADS_PER_BLOCK 256
#define BLOCKS_NUM 32
#define REPS 100000000

__device__ unsigned int histogram[HISTOGRAM_SIZE];

__global__ void spin(unsigned long int reps) {
  __shared__ unsigned int sharedHistogram[HISTOGRAM_SIZE];

  // Init shared memory
  for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
    sharedHistogram[i] = 0;
  }

  // Sync threads
  asm volatile("bar.sync 0;");

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int bdim = blockDim.x;

  for (unsigned int i = 0; i < reps; i++) {
    unsigned int t1, t2, diff, bin;

    // Sync threads
    asm volatile("bar.sync 0;");

    // Query clock reg & calc difference
    asm volatile(
        "mov.u32 %0, %%clock;\n\t"
        "mov.u32 %1, %%clock;\n\t"
        "sub.u32 %2, %1, %0;\n\t"
        : "=r"(t1), "=r"(t2), "=r"(diff)
        :
        : "memory");

    // log_2 bining using clz and diff
    asm volatile("clz.b32 %0, %1;\n\t" : "=r"(bin) : "r"(diff));

#ifdef DEBUG
    if (tid == 0 && i == 0) {
      printf("t1: %u, t2: %u, diff: %u\n", t1, t2, diff);
    }
#endif

    // Atomic add to shared histogram
    asm volatile("atom.shared.add.u32 %0[%1], 1;\n\t"
                 :
                 : "l"(sharedHistogram), "r"(bin * 4) // bytes->bits
                 : "memory");
  }

  // Sync threads
  asm volatile("bar.sync 0;");

  // Move shared into global mem
  for (int i = tid; i < HISTOGRAM_SIZE; i += bdim) {
    unsigned int val = sharedHistogram[i];
    asm volatile("atom.global.add.u32 %0[%1], %2;\n\t"
                 :
                 : "l"(histogram), "r"(i * 4), "r"(val)
                 : "memory");
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
    // Ripped from "Querying Device Properties" in
    // https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
    printf("Device Number: %d\n", i);
    printf("\tName: %s\n", prop.name);
    printf("\tMemory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("\tPeak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    // END of copied code
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

int sizeCheck(void) {
  size_t i_size = sizeof(unsigned int);
  size_t l_size = sizeof(unsigned long int);

  if (i_size != 4) {
    printf("Unsigned int appears to be %u bits on this system, spin_cycle only support 32 bit unsigned ints. Exiting", i_size);
    return 0;
  }
  else if (l_size != 8) {
    printf("Unsigned long int appears to be %u bits on this system, spin_cycle only support 32 bit unsigned long ints. Exiting", l_size);
    return 0;
  } 
  return 1;
}

int main() {
  if(sizeCheck() != 1) return;

  unsigned long int reps = REPS;
  runHistogram(reps);
  return 1;
}
