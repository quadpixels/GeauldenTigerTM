#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "device_launch_parameters.h"

extern __device__ int GetThdID();

__device__ int* g_pingpong_mailbox;

__global__ void pingpong() {
  const int tid = GetThdID();
  
  volatile int* mailbox = g_pingpong_mailbox + tid;

  // Wait on host wake
  bool done = false;
  int curr_timestamp;
  while (!done) {
    if (*mailbox != 0) {
      done = true;
      curr_timestamp = *(g_pingpong_mailbox + blockDim.x * gridDim.x);
      *mailbox = curr_timestamp;
    }
  }
  printf("[CUDA Kernel] Thread %d awakened at timestamp %d\n", tid, curr_timestamp);
}

__global__ void pingpong_signal(int x) {
  int* mailbox = g_pingpong_mailbox + x;
  *mailbox = 1;
}