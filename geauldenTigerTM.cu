#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

#include <iostream>
#include <memory>
#include <string>
#include <assert.h>

#include "device_launch_parameters.h"

// Caution: sizeof(long) may be 4

// "USE_PSTM" is defined in project settings
#define USE_PSTM

#ifdef USE_PSTM
  #include "pstm.h"
  #define SET_TXN_STATE(s) { atomicExch((int*)(&g_txnstate[tid]), (int)s); }
  #define INCREMENT_ABORT_COUNT { if (g_txnstate[tid] == ABORTED) { atomicAdd(g_n_aborts, 1); } }
  #define TX_READ(addr, ptr)      { if (!TxRead(tid, &aborted, (int*)(addr), (int*)(ptr), p_rwlog))       goto retry; }
  #define TX_READLONG(addr, ptr)  { if (!TxReadLong(tid, &aborted, (long*)(addr), (long*)(ptr), p_rwlog)) goto retry; }
  #define TX_WRITE(addr, val)     { if (!TxWrite(tid, &aborted, (int*)(addr), (int)(val), p_rwlog))       goto retry; }
  #define TX_WRITELONG(addr, val) { if (!TxWriteLong(tid, &aborted, (long*)(addr), (long)(val), p_rwlog)) goto retry; }
  #define TX_COMMIT { TxCommit(tid, &aborted, p_rwlog); }
  // Book-keeping stuff for PSTM
  extern __device__           int* g_se; // SE means Shadow Entry
  extern __device__ enum TxnState* g_txnstate;
  extern __device__           int* g_locks;
  extern __device__ int g_num_blk, g_num_thd_per_blk;
#endif

#define CE(call) {\
	call; \
	cudaError_t err = cudaGetLastError(); \
	if(err != cudaSuccess) { \
		printf("%s\n", cudaGetErrorString(err)); \
		assert(0); \
	} \
}

__device__ int* g_n_commits, * g_n_aborts;

// Workloads
extern __global__ void Hello(); // This is okay; extern __device__ is not okay
extern __global__ void counterTest(class RWLogs*, int*);
extern __global__ void counterTestLong(class RWLogs* rwlogs, int64_t* scratch);
extern __global__ void counterTestMultiple(class RWLogs* rwlogs, int* scratch, const int N);
extern __global__ void counterTestMultipleLong(class RWLogs* rwlogs, int64_t* scratch, const int N);

__device__ int GetThdID() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  int NB = 4, NT = 32;

  int run_mode = 2;
  for (int i=1; i<argc; i++) {
    int x;
    if (1 == sscanf(argv[i], "exp=%d", &x)) {
      printf("Run mode set to %d\n", x);
      run_mode = x;
    } else if (2 == sscanf(argv[i], "dim=%d,%d", &NB, &NT)) {
      printf("Dimension set to <<<%d, %d>>>\n", NB, NT);
    }
  }

  // Metadata for Orec-based STM algorithms
  #if defined(USE_PSTM)
  int* d_se, * d_locks;
  // Shadow entry
  CE(cudaMalloc(&d_se, sizeof(int) * SIZE));
  CE(cudaMemset(d_se, 0xFF, sizeof(int) * SIZE));
  CE(cudaMemcpyToSymbol(g_se, &d_se, sizeof(int*)));

  // Shadow entry lock
  CE(cudaMalloc(&d_locks, sizeof(int) * SIZE));
  CE(cudaMemset(d_locks, 0x00, sizeof(int) * SIZE));
  CE(cudaMemcpyToSymbol(g_locks, &d_locks, sizeof(int*)));
#if defined(USE_ESTM_UNDOLOG) || defined(USE_ESTM) || defined(USE_ESTM_COALESCED)
  int* d_readers;
  CE(cudaMalloc(&d_readers, sizeof(int) * SIZE * MAX_READERS_PER_ADDR));
  CE(cudaMemset(d_readers, 0xFF, sizeof(int) * SIZE * MAX_READERS_PER_ADDR));
  CE(cudaMemcpyToSymbol(g_readers, &d_readers, sizeof(int*)));
#endif
  enum TxnState* d_txnstate;
  CE(cudaMalloc(&d_txnstate, sizeof(enum TxnState) * MAX_CONCURRENT_TXN));
  CE(cudaMemcpyToSymbol(g_txnstate, &d_txnstate, sizeof(enum TxnState*)));

  // Read-write logs
  class RWLogs* h_rwlogs = new RWLogs[NB * NT];
  class RWLogs* d_rwlogs;
  CE(cudaMalloc(&d_rwlogs, sizeof(class RWLogs) * NB * NT));
  CE(cudaMemcpy(d_rwlogs, h_rwlogs, sizeof(RWLogs) * NB * NT, cudaMemcpyHostToDevice));

  // Commit and abort count
  int* d_n_commits, *d_n_aborts;
  CE(cudaMalloc(&d_n_commits, sizeof(int)));
  CE(cudaMalloc(&d_n_aborts, sizeof(int)));
  CE(cudaMemset(d_n_commits, 0x0, sizeof(int)));
  CE(cudaMemset(d_n_aborts, 0x0, sizeof(int)));
  CE(cudaMemcpyToSymbol(g_n_commits, &d_n_commits, sizeof(int*), 0, cudaMemcpyHostToDevice));
  CE(cudaMemcpyToSymbol(g_n_aborts, &d_n_aborts, sizeof(int*), 0, cudaMemcpyHostToDevice));

  #endif

  Hello<<<1, 1>>>();

  const int NUM_COUNTERS = 10;

  switch (run_mode) {
    case 0: {
      int* d_scratch, h_scratch;
      CE(cudaMalloc(&d_scratch, sizeof(int)));
      CE(cudaMemset(d_scratch, 0x00, sizeof(int)));
      counterTest<<<NB, NT>>>(d_rwlogs, d_scratch);
      CE(cudaMemcpy(&h_scratch, d_scratch, sizeof(int), cudaMemcpyDeviceToHost));
      printf("(int) Counter=%d\n", h_scratch);
      break;
    }
    case 10: {
      int64_t* d_scratch, h_scratch;
      CE(cudaMalloc(&d_scratch, sizeof(int64_t)));
      CE(cudaMemset(d_scratch, 0x00, sizeof(int64_t)));
      counterTestLong<<<NB, NT>>>(d_rwlogs, d_scratch);
      CE(cudaMemcpy(&h_scratch, d_scratch, sizeof(int64_t), cudaMemcpyDeviceToHost));
      printf("(long) Counter=%ld\n", h_scratch);
      break;
    }
    case 2: {
      int* d_scratch, h_scratch[NUM_COUNTERS];
      CE(cudaMalloc(&d_scratch, sizeof(int)*NUM_COUNTERS));
      CE(cudaMemset(d_scratch, 0x00, sizeof(int)*NUM_COUNTERS));
      counterTestMultiple<<<NB, NT>>>(d_rwlogs, d_scratch, NUM_COUNTERS);
      CE(cudaMemcpy(h_scratch, d_scratch, sizeof(int)*NUM_COUNTERS, cudaMemcpyDeviceToHost));
      printf("(multiple int's)");
      for (int i=0; i<NUM_COUNTERS; i++) {
        printf(" %d", h_scratch[i]);
      }
      printf("\n");
      break;
    }
    case 20: {
      int64_t* d_scratch, h_scratch[NUM_COUNTERS];
      CE(cudaMalloc(&d_scratch, sizeof(int64_t)*NUM_COUNTERS));
      CE(cudaMemset(d_scratch, 0x00, sizeof(int64_t)*NUM_COUNTERS));
      counterTestMultipleLong<<<NB, NT>>>(d_rwlogs, d_scratch, NUM_COUNTERS);
      CE(cudaMemcpy(h_scratch, d_scratch, sizeof(int64_t)*NUM_COUNTERS, cudaMemcpyDeviceToHost));
      printf("(multiple long's)");
      for (int i=0; i<NUM_COUNTERS; i++) {
        printf(" %ld", h_scratch[i]);
      }
      printf("\n");
      break;
    }
  }

  // Print statistics
  int h_n_commits, h_n_aborts;
  CE(cudaMemcpy(&h_n_commits, d_n_commits, sizeof(int), cudaMemcpyDeviceToHost));
  CE(cudaMemcpy(&h_n_aborts, d_n_aborts, sizeof(int), cudaMemcpyDeviceToHost));
  printf("%d commits, %d aborts\n", h_n_commits, h_n_aborts);

  exit(EXIT_SUCCESS);
}
