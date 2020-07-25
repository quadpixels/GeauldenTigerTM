#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>
#include <assert.h>

#include "device_launch_parameters.h"

#define USE_PSTM

#ifdef USE_PSTM
  #include "pstm.cu" // This file should be set to "not involved in the compilation process"
  #define SET_TXN_STATE(s) { atomicExch((int*)(&g_txnstate[tid]), (int)s); }
  #define INCREMENT_ABORT_COUNT { if (g_txnstate[tid] == ABORTED) { atomicAdd(g_n_aborts, 1); } }
  #define TX_READ(addr, ptr)      { if (!TxRead(tid, &aborted, (int*)(addr), (int*)(ptr), p_rwlog))       goto retry; }
  #define TX_READLONG(addr, ptr)  { if (!TxReadLong(tid, &aborted, (long*)(addr), (long*)(ptr), p_rwlog)) goto retry; }
  #define TX_WRITE(addr, val)     { if (!TxWrite(tid, &aborted, (int*)(addr), (int)(val), p_rwlog))       goto retry; }
  #define TX_WRITELONG(addr, val) { if (!TxWriteLong(tid, &aborted, (long*)(addr), (long)(val), p_rwlog)) goto retry; }
  #define TX_COMMIT { TxCommit(tid, &aborted, p_rwlog); }
#endif

#define CE(call) {\
	call; \
	cudaError_t err = cudaGetLastError(); \
	if(err != cudaSuccess) { \
		printf("%s\n", cudaGetErrorString(err)); \
		assert(0); \
	} \
}

__global__ void hello(int* x) {
  printf("Hello from thd (%d,%d), g_se=%p\n", blockIdx.x, threadIdx.x, g_se);
  *x = 10;
}

__device__ int GetThdID() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void counterTest(class RWLogs* rwlogs, int* scratch) {
  __syncthreads();
  const int tid = GetThdID();
  RWLogs* p_rwlog = &(rwlogs[tid]);
  SET_TXN_STATE(RUNNING);
  p_rwlog->init();
  __threadfence();
  int attempt = 0;
  const int ATTEMPT_LIMIT = 1000000;
  printf("Thd %d start\n", tid);
retry:
  p_rwlog->releaseAll(tid);
  p_rwlog->init();
  if (attempt++ >= ATTEMPT_LIMIT) { return; }
  
  INCREMENT_ABORT_COUNT;
  SET_TXN_STATE(RUNNING);
  bool aborted = false;
  int c;
  TX_READ(scratch, &c);
  TX_WRITE(scratch, c + 1);
  TX_COMMIT;
  if (aborted) goto retry;
  else SET_TXN_STATE(ABORTED);

  printf("Tx %d counter %d->%d\n", tid, c, c + 1);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  int NB = 4, NT = 32;

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

  int* d_scratch;
  CE(cudaMalloc(&d_scratch, sizeof(int)));
  CE(cudaMemset(d_scratch, 0x00, sizeof(int)));
  counterTest<<<NB, NT>>>(d_rwlogs, d_scratch);



  // Print statistics
  int h_n_commits, h_n_aborts;
  CE(cudaMemcpy(&h_n_commits, d_n_commits, sizeof(int), cudaMemcpyDeviceToHost));
  CE(cudaMemcpy(&h_n_aborts, d_n_aborts, sizeof(int), cudaMemcpyDeviceToHost));
  printf("%d commits, %d aborts\n", h_n_commits, h_n_aborts);

  exit(EXIT_SUCCESS);
}
