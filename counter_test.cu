#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "pstm.h"

#include "device_launch_parameters.h"

#define USE_PSTM

#ifdef USE_PSTM
  #define SET_TXN_STATE(s) { atomicExch((int*)(&g_txnstate[tid]), (int)s); }
  #define INCREMENT_ABORT_COUNT { if (g_txnstate[tid] == ABORTED) { atomicAdd(g_n_aborts, 1); } }
  #define TX_READ(addr, ptr)      { if (!TxRead(tid, &aborted, (int*)(addr), (int*)(ptr), p_rwlog))       goto retry; }
  #define TX_READLONG(addr, ptr)  { if (!TxReadLong(tid, &aborted, (long*)(addr), (long*)(ptr), p_rwlog)) goto retry; }
  #define TX_WRITE(addr, val)     { if (!TxWrite(tid, &aborted, (int*)(addr), (int)(val), p_rwlog))       goto retry; }
  #define TX_WRITELONG(addr, val) { if (!TxWriteLong(tid, &aborted, (long*)(addr), (long)(val), p_rwlog)) goto retry; }
  #define TX_COMMIT { TxCommit(tid, &aborted, p_rwlog); }
#endif

// Note: Enable "-rdc" in Visual Studio for functions in other translation units to work
// Project Properties --> CUDA C/C++ --> Common --> Generate Relocatable Device Code
extern __device__ enum TxnState* g_txnstate;
extern __device__ int* g_n_commits, * g_n_aborts;
extern __device__ bool TxRead(const int tid, bool* aborted,	const int* const addr, int* const value, class RWLogs* rwlog);
extern __device__ bool TxWrite(const int tid, bool* aborted,	int* const addr, const int value, class RWLogs* rwlog);
extern __device__ bool TxCommit(const int tid, bool* aborted, class RWLogs* rwlog);
extern __device__ int GetThdID();

__global__ void Hello() {
  printf("Hello\n");
  __syncthreads();
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
retry:
  p_rwlog->releaseAll(tid);
  p_rwlog->init();
  if (attempt++ >= ATTEMPT_LIMIT) { return; }

  INCREMENT_ABORT_COUNT;
  SET_TXN_STATE(RUNNING);
  bool aborted = false;
  int c;
  //TX_READ(scratch, &c);
  { if (!TxRead(tid, &aborted, (int*)scratch, (int*)&c, p_rwlog)) goto retry; }
  TX_WRITE(scratch, c + 1);
  TX_COMMIT;
  if (aborted) goto retry;
  else SET_TXN_STATE(ABORTED);
}

__global__ void counterTestLong(class RWLogs* rwlogs, long* scratch) {
  __syncthreads();
  const int tid = GetThdID();
  RWLogs* p_rwlog = &(rwlogs[tid]);
  SET_TXN_STATE(RUNNING);
  p_rwlog->init();
  __threadfence();
  int attempt = 0;
  const int ATTEMPT_LIMIT = 1000000;
retry:
  p_rwlog->releaseAll(tid);
  p_rwlog->init();
  if (attempt++ >= ATTEMPT_LIMIT) { return; }

  INCREMENT_ABORT_COUNT;
  SET_TXN_STATE(RUNNING);
  bool aborted = false;
  int c;
  TX_READLONG(scratch, &c);
  TX_WRITELONG(scratch, c + 1);
  TX_COMMIT;
  if (aborted) goto retry;
  else SET_TXN_STATE(ABORTED);
}

__global__ void counterTestMultiple(class RWLogs* rwlogs, int* scratch, const int N) {
  const int tid = GetThdID();
  RWLogs* p_rwlog = &(rwlogs[tid]);
  SET_TXN_STATE(RUNNING);
  p_rwlog->init();
  __threadfence();
  int attempt = 0;
  const int ATTEMPT_LIMIT = 1000000;
retry:
  p_rwlog->releaseAll(tid);
  p_rwlog->init();
  if (attempt++ >= ATTEMPT_LIMIT) { return; }
  bool aborted = false;

  INCREMENT_ABORT_COUNT;
  SET_TXN_STATE(RUNNING);
  for (int n=0; n<N; n++) {
    const int idx = (n + tid) % N;
    int c;
    TX_READ(&(scratch[idx]), &c);
    TX_WRITE(&(scratch[idx]), c+1);
  }
  TX_COMMIT;
  if (aborted) goto retry;
  else SET_TXN_STATE(ABORTED);
}