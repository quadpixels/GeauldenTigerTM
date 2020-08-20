// PSTM is largely built on the descriptions in the paper "PR-STM: Priority Rule Based Software Transactions for the GPU"
// by Qi Shen, Craig Sharp, William Blewitt, Gary Ushaw, Euro-Par 2015

#ifndef __PSTM_H
#define __PSTM_H

// The following includes are needed for Windows
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define LOG_LENGTH 500
#define SIZE 1048576
enum TxnState {
	RUNNING,
	ABORTED,
	COMMITTING,
	ROLLBACK,
	COMMITTED,
};
#define MAX_CONCURRENT_TXN 23040

#define ENABLE_STATISTICS

// Do not distinguish between reads and writes
class RWLogs {
	int owned[LOG_LENGTH];
	int nowned;
public:
	RWLogs() { init(); }
	__host__ __device__ void init() { nowned = nwrites = 0; }
	__device__ void appendShadowEntry(const int my_tid, int* p_shadow_entry);
	__device__ void releaseAll(const int my_tid);
	__device__ bool validate(const int my_tid);
	int* write_addrs[LOG_LENGTH];
	int write_values[LOG_LENGTH];
	int nwrites;
	__device__ void appendWrite(int* const addr, int val);
	__device__ void commitToMemory();
	__device__ bool findValueInWriteLog(const int* addr, int* val);
	__device__ void Dump();
};

// Prototypes
extern __device__ bool TxRead(const int tid, bool* aborted,	const int* const addr, int* const value, class RWLogs* rwlog);
extern __device__ bool TxReadLong(const int tid, bool* aborted,	const int64_t* const addr, int64_t* const value, RWLogs* rwlog);
extern __device__ bool TxWrite(const int tid, bool* aborted,	int* const addr, const int value, class RWLogs* rwlog);
extern __device__ bool TxWriteLong(const int tid, bool* aborted, int64_t* const addr, const int64_t value, RWLogs* rwlog);
extern __device__ bool TxCommit(const int tid, bool* aborted, class RWLogs* rwlog);

#endif