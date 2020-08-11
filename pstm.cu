#include "pstm.h"
#include <cuda_runtime.h>

__device__           int* g_se; // SE means Shadow Entry
__device__ enum TxnState* g_txnstate;
__device__           int* g_locks;
__device__ int g_num_blk, g_num_thd_per_blk;

// book-keeping for all TMs
extern __device__ int* g_n_commits, * g_n_aborts;

__device__ RWLogs* g_rwlogs;

__device__ void RWLogs::releaseAll(const int my_tid)  {
	for (int i = 0; i < nowned; i++) {
		atomicCAS(&(g_se[owned[i]]), my_tid, 0xFFFFFFFF); // May be preempted, so we use atomicCAS
																											// atomicExch(&(g_se[owned[i]]), 0xFFFFFFFF);
																											// This is wrong, will cause over-abort.
		owned[i] = -999;
	}
	nowned = 0;
}

__device__ bool RWLogs::validate(const int my_tid) {
	for (int i = 0; i < nowned; i++) {
			const int owner_tid = g_se[owned[i]];
			if (owner_tid != my_tid) {
				return false;
		}
	}
	return true;
}

__device__ void RWLogs::appendWrite(int* const addr, int val) {
	if (nwrites >= LOG_LENGTH) {
		asm("brkpt;");
	}
	write_addrs[nwrites] = addr;
	write_values[nwrites] = val;
	nwrites++;
}

__device__ bool RWLogs::findValueInWriteLog(const int* addr, int* val) {
	for (int i = nwrites - 1; i >= 0; i--) {
		if (addr == write_addrs[i]) {
			*val = write_values[i]; return true;
		}
	}
	return false;
}

__device__ void RWLogs::commitToMemory() {
	for (int i = 0; i < nwrites; i++) {
		int* addr = write_addrs[i];
		int  val = write_values[i];
		atomicExch(addr, val);
	}
}

__device__ void RWLogs::appendShadowEntry(const int my_tid, int* p_shadow_entry) {
	if (nowned >= LOG_LENGTH) {
#if __CUDA_ARCH__ >= 2000
		printf("Running out of shadow entry spacce!");
#endif
		asm("brkpt;");
	}
	int offset = p_shadow_entry - g_se;
	for (int i = 0; i < nowned; i++) { // Dedup.
		if (owned[i] == offset) return;
	}
	owned[nowned++] = offset;
}

__device__ void RWLogs::Dump() {
	printf("owned=%d\n", nowned);
}

__device__ bool TxAcquire(const int tid, int* p_se/* ptr to shadow entry */) {
	bool aborted = false;
	int shadow = *p_se;
	if (shadow == tid) { return false; }
	else if (shadow != 0xFFFFFFFF) { // solves livelocks.
		if (shadow > tid) {
			if (g_txnstate[shadow] == ABORTED) {
				int bleh = atomicCAS(p_se, shadow, tid);
				if (bleh == shadow) {}
				else { aborted = true; }
			}
			else if (atomicCAS((int*)&g_txnstate[shadow], (int)RUNNING, (int)ABORTED) == (int)RUNNING) {
				int bleh = atomicCAS(p_se, shadow, tid);
				if (bleh == shadow) {}
				else { aborted = true; }
			}
			else { aborted = true; }
		}
		else { aborted = true; }
	}
	else if (shadow != tid) {
		if (atomicCAS(p_se, 0xFFFFFFFF, tid) == 0xFFFFFFFF) {}
		else { aborted = true; }
	}
	return aborted;
}

// Non-class-function calls
// If success, return TRUE, else return FALSE
__device__ bool TxRead(const int tid, bool* aborted,
	const int* const addr, int* const value, class RWLogs* rwlog) {
	if (*aborted == true) return false;
	int64_t lock_idx = ((int64_t)addr / 16L) % SIZE;
	(*aborted) |= TxAcquire(tid, &g_se[lock_idx]);
	if (*aborted == false) {
		rwlog->appendShadowEntry(tid, &g_se[lock_idx]);
		if (!(rwlog->findValueInWriteLog(addr, value)))
			*value = *addr;
	}
	return !(*aborted);
}

__device__ bool TxWrite(const int tid, bool* aborted,
	int* const addr, const int value, class RWLogs* rwlog) {
	if (*aborted == true) return !(*aborted);
	int64_t lock_idx = ((int64_t)addr / 16) % SIZE;
	(*aborted) |= TxAcquire(tid, &g_se[lock_idx]);
	if (*aborted == false) {
		rwlog->appendShadowEntry(tid, &g_se[lock_idx]);
		rwlog->appendWrite(addr, value);
	}
	return !(*aborted);
}

__device__ bool TxReadLong(const int tid, bool* aborted,
	const int64_t* const addr, int64_t* const value, RWLogs* rwlog) {
	int lower, upper;
	int* addr_lower = (int*)addr, * addr_upper = addr_lower + 1;
	if (!TxRead(tid, aborted, addr_lower, &lower, rwlog)) return !(*aborted);
	if (!TxRead(tid, aborted, addr_upper, &upper, rwlog)) return !(*aborted);
	int64_t ret = ((int64_t)(upper) << 32) | lower;
	*value = ret;
	return !(*aborted);
}

__device__ bool TxWriteLong(const int tid, bool* aborted,
	int64_t* const addr, const int64_t value, RWLogs* rwlog) {
	int lower = (value & 0xFFFFFFFF), upper = ((int64_t)value >> 32) & 0xFFFFFFFF;
	int* addr_lower = (int*)addr, * addr_upper = addr_lower + 1;
	if (!TxWrite(tid, aborted, addr_lower, lower, rwlog)) return false;
	if (!TxWrite(tid, aborted, addr_upper, upper, rwlog)) return false;
	return !(*aborted);
}

__device__ bool TxCommit(const int tid, bool* aborted, class RWLogs* rwlog) {
	if (*aborted == false) {
		if (atomicCAS((int*)&g_txnstate[tid], (int)RUNNING, (int)COMMITTING) == RUNNING) {
			if (true || rwlog->validate(tid)) { // No need to validate
				int num_commits = atomicAdd(g_n_commits, 1);
				rwlog->commitToMemory();
			}
			else {
				*aborted = true;
			}
		}
		else {
			*aborted = true;
			atomicAdd(g_n_aborts, 1);
		}
	}
	rwlog->releaseAll(tid);
	return !(*aborted);
}