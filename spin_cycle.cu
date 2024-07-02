#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#define HISTOGRAM_SIZE 32  // clock values are 32 bit
#define THREADS_PER_BLOCK 256
#define BLOCKS_NUM 32
#define REPS 10000000

//namespace cg = cooperative_groups;

//__device__ unsigned int histogram[HISTOGRAM_SIZE];

__global__ void spin(unsigned long int reps, unsigned long long int * d_histogram) {
  //cg::thread_block block = cg::this_thread_block();
  __shared__ unsigned long long sharedHistogram[HISTOGRAM_SIZE];

  // Init shared memory
  for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
    sharedHistogram[i] = 0;
  }

  // Sync threads
  asm volatile("bar.sync 0;");

  unsigned int tid = threadIdx.x;
  unsigned int bdim = blockDim.x;

  for (unsigned long int i = 0; i < reps; i++) {
    unsigned int t1, t2, diff, bin;

    // Sync threads
    asm volatile("bar.sync 0;");

    // Query clock reg & calc difference
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(t1));

/*#ifdef DEBUG
    int churn = 0;
    for (int x = 0; x < 100; x++) {
	churn = churn * x;
    }
#endif
*/
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(t2));

    diff = (t2 >= t1) ? (t2 - t1) : (0xFFFFFFFF - t1 + t2 + 1);
    // log_2 bining using clz and diff
    bin = __clz(diff);


#ifdef DEBUG
    if (tid == 0 && i < 10) {
	printf("Iteration %lu: t1: %u, t2: %u, diff: %u, bin: %u\n", i, t1, t2, diff, bin);
    }
#endif

    // Atomic add to shared histogram
    atomicAdd(&sharedHistogram[bin], 1);
  }

  // Sync threads
  asm volatile("bar.sync 0;");

  // Move shared into global mem
	for (int i = tid; i < HISTOGRAM_SIZE; i += bdim) {
	  unsigned int val = sharedHistogram[i];
	  atomicAdd(&d_histogram[i], val);
	}

}

// Runner
void runHistogram(unsigned long int reps) {

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
    printf("\t# repetitions %lu\t", reps);

    // Reset (Not sure if this is needed)
    cudaDeviceSynchronize();
    unsigned long long int *d_histogram;
	cudaMalloc(&d_histogram, HISTOGRAM_SIZE * sizeof(unsigned long long int));
	cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned long long int));

	// Pass d_histogram to the kernel
	spin<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(reps, d_histogram);

	// Copy results back to host
	unsigned long long int h_histogram[HISTOGRAM_SIZE] = {0};
	cudaMemcpy(h_histogram, d_histogram, HISTOGRAM_SIZE * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_histogram);



    // Print Results
    unsigned long long total = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
      printf("Bin %d: %llu\n", i, h_histogram[i]);
    }

	printf("Total measurements: %llu\n", total);
	printf("Expected measurements: %llu\n", (unsigned long long)reps * BLOCKS_NUM * THREADS_PER_BLOCK);

  }
}

int sizeCheck(void) {
  size_t i_size = sizeof(unsigned int);
  size_t l_size = sizeof(unsigned long int);

  if (i_size != 4) {
    printf("Unsigned int appears to be %zu bits on this system, spin_cycle only support 32 bit unsigned ints. Exiting", i_size);
    return 0;
  }
  else if (l_size != 8) {
    printf("Unsigned long int appears to be %zu bits on this system, spin_cycle only support 32 bit unsigned long ints. Exiting", l_size);
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
