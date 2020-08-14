#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "pstm.h"

#include "device_launch_parameters.h"

#include "linkedlist.h"


#define USE_PSTM

#ifdef USE_PSTM
#define SET_TXN_STATE(s) { atomicExch((int*)(&g_txnstate[tid]), (int)s); }
#define INCREMENT_ABORT_COUNT { if (g_txnstate[tid] == ABORTED) { atomicAdd(g_n_aborts, 1); } }
#define TX_READ(addr, ptr)      { if (!TxRead(tid, &aborted, (int*)(addr), (int*)(ptr), p_rwlog))       goto retry; }
#define TX_READLONG(addr, ptr)  { if (!TxReadLong(tid, &aborted, (int64_t*)(addr), (int64_t*)(ptr), p_rwlog)) goto retry; }
#define TX_WRITE(addr, val)     { if (!TxWrite(tid, &aborted, (int*)(addr), (int)(val), p_rwlog))       goto retry; }
#define TX_WRITELONG(addr, val) { if (!TxWriteLong(tid, &aborted, (int64_t*)(addr), (int64_t)(val), p_rwlog)) goto retry; }
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

__global__ void listbmk_GPU_serial(ListNode* list_head, ListNode* new_nodes, int count) {
	const int tidx = GetThdID();
	ListNode* const BASE = list_head;
	if (tidx > 0) return;
	for (int i=0; i<count; i++) {
		ListNode* my_node = &(new_nodes[i]);
		ListNode* curr = list_head, *prev = NULL;
		int val = my_node->val;
		while (val > curr->val) {
			prev = curr;
			curr = &(BASE[curr->next_idx]);
		}
		prev->next_idx    = my_node - BASE;
		my_node->next_idx = curr - BASE;
	}
}