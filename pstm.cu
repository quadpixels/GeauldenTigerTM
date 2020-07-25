#include "pstm.h"

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
	long lock_idx = ((int64_t)addr / 16L) % SIZE;
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
	long lock_idx = ((int64_t)addr / 16) % SIZE;
	(*aborted) |= TxAcquire(tid, &g_se[lock_idx]);
	if (*aborted == false) {
		rwlog->appendShadowEntry(tid, &g_se[lock_idx]);
		rwlog->appendWrite(addr, value);
	}
	return !(*aborted);
}

__device__ bool TxReadLong(const int tid, bool* aborted,
	const long* const addr, long* const value, RWLogs* rwlog) {
	int lower, upper;
	int* addr_lower = (int*)addr, * addr_upper = addr_lower + 1;
	if (!TxRead(tid, aborted, addr_lower, &lower, rwlog)) return !(*aborted);
	if (!TxRead(tid, aborted, addr_upper, &upper, rwlog)) return !(*aborted);
	long ret = ((int64_t)(upper) << 32) | lower;
	*value = ret;
	return !(*aborted);
}

__device__ bool TxWriteLong(const int tid, bool* aborted,
	long* const addr, const long value, RWLogs* rwlog) {
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
