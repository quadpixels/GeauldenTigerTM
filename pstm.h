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
__device__           int* g_se; // SE means Shadow Entry
__device__ enum TxnState* g_txnstate;
__device__           int* g_locks;
__device__ int g_num_blk, g_num_thd_per_blk;

#define ENABLE_STATISTICS

#ifdef ENABLE_STATISTICS
__device__ int* g_n_commits, * g_n_aborts;
#endif
class RWLogs;
__device__ RWLogs* g_rwlogs;

__device__ int g_num_committers, * g_num_committers_log, g_num_committers_idx;

// Do not distinguish between reads and writes
class RWLogs {
	int owned[LOG_LENGTH];
	int nowned;
public:
	RWLogs() { init(); }
	__host__ __device__ void init() { nowned = nwrites = 0; }
	__device__ void appendShadowEntry(const int my_tid, int* p_shadow_entry);
	__device__ void releaseAll(const int my_tid) {
		for (int i = 0; i < nowned; i++) {
			atomicCAS(&(g_se[owned[i]]), my_tid, 0xFFFFFFFF); // May be preempted, so we use atomicCAS
//			atomicExch(&(g_se[owned[i]]), 0xFFFFFFFF); // This is wrong, will cause over-abort.
			owned[i] = -999;
		}
		nowned = 0;
	}
	__device__ bool validate(const int my_tid) {
		for (int i = 0; i < nowned; i++) {
			const int owner_tid = g_se[owned[i]];
			if (owner_tid != my_tid) {
				return false;
			}
		}
		return true;
	}
	int* write_addrs[LOG_LENGTH];
	int write_values[LOG_LENGTH];
	int nwrites;
	__device__ void appendWrite(int* const addr, int val) {
		if (nwrites >= LOG_LENGTH) {
			asm("brkpt;");
		}
		write_addrs[nwrites] = addr;
		write_values[nwrites] = val;
		nwrites++;
	}
	__device__ void commitToMemory() {
		for (int i = 0; i < nwrites; i++) {
			int* addr = write_addrs[i];
			int  val = write_values[i];
			atomicExch(addr, val);
		}
	}

	__device__ bool findValueInWriteLog(const int* addr, int* val) {
		for (int i = nwrites - 1; i >= 0; i--) {
			if (addr == write_addrs[i]) {
				*val = write_values[i]; return true;
			}
		}
		return false;
	}
};

// Prototypes
__inline__ __device__ bool TxRead(const int tid, bool* aborted,
	const int* const addr, int* const value, class RWLogs* rwlog, int* reason);
__device__ bool TxWrite(const int tid, bool* aborted,
	int* const addr, const int value, class RWLogs* rwlog, int* reason);
__device__ bool TxCommit(const int tid, bool* aborted, class RWLogs* rwlog);

#endif