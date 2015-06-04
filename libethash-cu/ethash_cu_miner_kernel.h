#ifndef _ETHASH_CU_MINER_KERNEL_H_
#define _ETHASH_CU_MINER_KERNEL_H_

typedef unsigned long long int ulong;
typedef unsigned int  uint;

typedef union
{
	ulong ulongs[32 / sizeof(ulong)];
	uint uints[32 / sizeof(uint)];
} hash32_t;


typedef union
{
	ulong ulongs[64 / sizeof(ulong)];
	uint  uints[64 / sizeof(uint)];
} hash64_t;


typedef union
{
	uint uints[128 / sizeof(uint)]; 
} hash128_t;

//typedef uint hash128_t;

cudaError set_constants(
	uint * dag_size,
	uint * acceses,
	uint * max_outputs,
	uint * workgroup_size
);

void run_ethash_hash(
	hash32_t* g_hashes,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce
);

void run_ethash_search(
	uint search_batch_size,
	uint workgroup_size,
	cudaStream_t stream,
	uint* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target
);

#endif
