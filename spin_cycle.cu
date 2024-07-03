#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#define HISTOGRAM_SIZE 32  // clock values are 32 bit
#define THREADS_PER_BLOCK 256
#define BLOCKS_NUM 32
#define REPS 10000000


//__device__ unsigned int histogram[HISTOGRAM_SIZE];

__global__ void spin(unsigned long long int reps,
                     unsigned long long int *d_histogram) {
#ifdef DEBUG
  __shared__ volatile int s_mem[256];
#endif
  __shared__ unsigned long long sharedHistogram[HISTOGRAM_SIZE];

  // Init shared memory
  for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
    sharedHistogram[i] = 0;
  }

  // Sync threads
  asm volatile("bar.sync 0;");

  unsigned int tid = threadIdx.x;
  unsigned int bdim = blockDim.x;

  for (unsigned long long int i = 0; i < reps; i++) {
    unsigned int t1, t2, diff, bin;

    asm volatile("bar.sync 0;");
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));

#ifdef DEBUG
    // Dummy code that will introduce forced random variability
    if (tid % 2 == 0) {
      for (int j = 0; j < tid; j++) {
        s_mem[tid] = j * tid;
      }
    } else {
      for (int j = 0; j < 256 - tid; j++) {
        s_mem[tid] = j * (256 - tid);
      }
    }
#endif

    asm volatile("bar.sync 0;");
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t2));

    // Calc Clock delta and log_2 bin using clz
    diff = (t2 >= t1) ? (t2 - t1) : (0xFFFFFFFF - t1 + t2 + 1);
    bin = __clz(diff);

#ifdef DEBUG
    // Print a few samples
    if (threadIdx.x == 0 && blockIdx.x == 0 && i < 10) {
      printf("Debug: t1=%u, t2=%u, diff=%u, bin=%u\n", t1, t2, diff, bin);
    }
#endif

    // Add to shared histogram
    atomicAdd(&sharedHistogram[bin], 1ULL);
  }

  asm volatile("bar.sync 0;");

  // Move shared histogram into global histogram
  for (int i = tid; i < HISTOGRAM_SIZE; i += bdim) {
    atomicAdd(&d_histogram[i], sharedHistogram[i]);
  }
}

// Runs spin()
void runHistogram(unsigned long long int reps) {
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
    printf("\t# repetitions %llu\t", reps);

    cudaDeviceSynchronize();

    unsigned long long int *d_histogram;
    cudaMalloc(&d_histogram, HISTOGRAM_SIZE * sizeof(unsigned long long int));
    cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned long long int));

    // Pass d_histogram to the kernel
    spin<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(reps, d_histogram);

    // Copy results back to host
    unsigned long long int h_histogram[HISTOGRAM_SIZE] = {0};
    cudaMemcpy(h_histogram, d_histogram,
               HISTOGRAM_SIZE * sizeof(unsigned long long int),
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_histogram);

    // Print Results
    unsigned long long total = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
      total += (unsigned long long)h_histogram[i];
      printf("Bin %d: %llu\n", i, h_histogram[i]);
    }

    printf("Total measurements: %llu\n", total);
    printf("Expected measurements: %llu\n",
           (unsigned long long)reps * BLOCKS_NUM * THREADS_PER_BLOCK);
  }
}

int systemConfigValidation(void) {
  unsigned int warp_size;
  size_t ui_size, lu_size, llu_size;

  ui_size = sizeof(unsigned int);
  lu_size = sizeof(unsigned long int);
  llu_size = sizeof(unsigned long long int);
  //asm("mov.u32 %0, %warpsize;" : "=r"(warp_size));

  if (ui_size != 4) {
    printf(
        "Unsigned int appears to be %zu bits on this system, spin_cycle only "
        "support 32 bit unsigned ints. Exiting",
        ui_size);
    return 0;
  } else if (lu_size != 8) {
    printf(
        "Unsigned long int appears to be %zu bits on this system, spin_cycle "
        "only support 64 bit unsigned long ints. Exiting",
        lu_size);
    return 0;
  } else if (llu_size != 8) {
    printf(
        "Unsigned long long int appears to be %zu bits on this system, spin_cycle "
        "only support 128 bit unsigned long ints. Exiting",
        llu_size);
    return 0;
  }
  return 1;
}

int main() {
  if (systemConfigValidation() != 1) return 1;

  unsigned long long int reps = REPS;
  runHistogram(reps);
  return 1;
}
