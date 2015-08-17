/*
* Genoil's CUDA mining kernel for Ethereum
* based on Tim Hughes' opencl kernel.
* thanks to trpuvot,djm34,sp,cbuchner for things i took from ccminer.
*/

#define SHUFFLE_MIN_VER 350

#include "ethash_cu_miner_kernel.h"
#include "ethash_cu_miner_kernel_globals.h"
#include "rotl64.cuh"
#include "cuda_helper.h"
#include "keccak.cuh"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "vector_types.h"

#define ACCESSES 64
#define THREADS_PER_HASH (128 / 16)

#define FNV_PRIME	0x01000193

#define SWAP64(v) \
  ((ROTL64L(v,  8) & 0x000000FF000000FF) | \
   (ROTL64L(v, 24) & 0x0000FF000000FF00) | \
   (ROTL64H(v, 40) & 0x00FF000000FF0000) | \
   (ROTL64H(v, 56) & 0xFF000000FF000000))

#define PACK64(result, lo, hi) asm("mov.b64 %0, {%1,%2};//pack64"  : "=l"(result) : "r"(lo), "r"(hi));
#define UNPACK64(lo, hi, input) asm("mov.b64 {%0, %1}, %2;//unpack64" : "=r"(lo),"=r"(hi) : "l"(input));

#define copy(dst, src, count) for (uint32_t i = 0; i < count; i++) { (dst)[i] = (src)[i]; }

#define countof(x) (sizeof(x) / sizeof(x[0]))

#define fnv(x,y) ((x) * FNV_PRIME ^(y))

__device__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 c;
	c.x = a.x * FNV_PRIME ^ b.x;	
	c.y = a.y * FNV_PRIME ^ b.y;
	c.z = a.z * FNV_PRIME ^ b.z;
	c.w = a.w * FNV_PRIME ^ b.w;
	return c;
}

__device__ uint32_t fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}


__device__ hash64_t init_hash(hash32_t const* header, uint64_t nonce)
{
	hash64_t init;

	// sha3_512(header .. nonce)
	uint64_t state[25];

	copy(state, header->uint64s, 4);
	state[4] = nonce;
	state[5] = 0x0000000000000001;
	state[6] = 0;
	state[7] = 0;
	state[8] = 0x8000000000000000;
	for (uint32_t i = 9; i < 25; i++)
	{
		state[i] = 0;
	}
	
	keccak_f1600_block((uint2 *)state, 8);
	copy(init.uint64s, state, 8);
	return init;
}

__device__ uint32_t inner_loop(uint4 mix, uint32_t thread_id, uint32_t* share, hash128_t const* g_dag)
{
	// share init0
	if (thread_id == 0)
		*share = mix.x;

	uint32_t init0 = *share;
	
	uint32_t a = 0;

	do
	{
		
		bool update_share = thread_id == ((a >> 2) & (THREADS_PER_HASH-1));

		//#pragma unroll 4
		for (uint32_t i = 0; i < 4; i++)
		{

			if (update_share)
			{
				uint32_t m[4] = { mix.x, mix.y, mix.z, mix.w };
				*share = fnv(init0 ^ (a + i), m[i]) % d_dag_size;
			}
			__threadfence_block();

#if __CUDA_ARCH__ >= 350
			mix = fnv4(mix, __ldg(&g_dag[*share].uint4s[thread_id]));
#else
			mix = fnv4(mix, g_dag[*share].uint4s[thread_id]);
#endif
			
		}
		
	} while ((a += 4) != ACCESSES);
	
	return fnv_reduce(mix);
}

__device__ hash32_t final_hash(hash64_t const* init, hash32_t const* mix)
{
	uint64_t state[25];

	hash32_t hash;

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->uint64s, 8);
	copy(state + 8, mix->uint64s, 4);
	state[12] = 0x0000000000000001;
	for (uint32_t i = 13; i < 16; i++)
	{
		state[i] = 0;
	}
	state[16] = 0x8000000000000000;
	for (uint32_t i = 17; i < 25; i++)
	{
		state[i] = 0;
	}

	keccak_f1600_block((uint2 *)state, 1);

	// copy out
	copy(hash.uint64s, state, 4);
	return hash;
}

typedef union
{
	hash64_t init;	
	hash32_t mix;
} compute_hash_share;

#if __CUDA_ARCH__ >= SHUFFLE_MIN_VER
__device__ uint64_t compute_hash_shuffle(
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t nonce
	)
{
	// sha3_512(header .. nonce)
	uint64_t state[25];
	
	copy(state, g_header->uint64s, 4);
	state[4] = nonce;
	state[5] = 0x0000000000000001ULL;
	for (uint32_t i = 6; i < 25; i++)
	{
		state[i] = 0;
	}
	state[8] = 0x8000000000000000ULL;
	keccak_f1600_block((uint2 *)state, 8);

	// Threads work together in this phase in groups of 8.
	const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
	const int start_lane = threadIdx.x & ~(THREADS_PER_HASH - 1);
	const int mix_idx = (thread_id & 3); 

	uint4 mix;

	uint32_t shuffle[16];
	//uint32_t * init = (uint32_t *)state;

	uint32_t init[16];
	UNPACK64(init[0], init[1], state[0]);
	UNPACK64(init[2], init[3], state[1]);
	UNPACK64(init[4], init[5], state[2]);
	UNPACK64(init[6], init[7], state[3]);
	UNPACK64(init[8], init[9], state[4]);
	UNPACK64(init[10], init[11], state[5]);
	UNPACK64(init[12], init[13], state[6]);
	UNPACK64(init[14], init[15], state[7]);

	for (int i = 0; i < THREADS_PER_HASH; i++)
	{

		// share init among threads
		for (int j = 0; j < 16; j++)
			shuffle[j] = __shfl(init[j], start_lane + i);
		
		// ugly but avoids local reads/writes
		if (mix_idx == 0) {
			mix = make_uint4(shuffle[0], shuffle[1], shuffle[2], shuffle[3]);			
		}
		else if (mix_idx == 1) {
			mix = make_uint4(shuffle[4], shuffle[5], shuffle[6], shuffle[7]);
		}
		else if (mix_idx == 2) {
			mix = make_uint4(shuffle[8], shuffle[9], shuffle[10], shuffle[11]);
		}
		else {
			mix = make_uint4(shuffle[12], shuffle[13], shuffle[14], shuffle[15]);
		}
		
		uint32_t init0 = __shfl(shuffle[0], start_lane);
		
		
		for (uint32_t a = 0; a < ACCESSES; a+=4)
		{
			int t = ((a >> 2) & (THREADS_PER_HASH - 1));

			for (uint32_t b = 0; b < 4; b++)
			{
				if (thread_id == t)
				{
					shuffle[0] = fnv(init0 ^ (a + b), ((uint32_t *)&mix)[b]) % d_dag_size;;
				}
				shuffle[0] = __shfl(shuffle[0], start_lane + t);

				mix = fnv4(mix, g_dag[shuffle[0]].uint4s[thread_id]);			
			}
		} 

		uint32_t thread_mix = fnv_reduce(mix);

		// update mix accross threads

		for (int j = 0; j < 8; j++)
			shuffle[j] = __shfl(thread_mix, start_lane + j);

		if (i == thread_id) {	

			//move mix into state:
			PACK64(state[8],  shuffle[0], shuffle[1]);
			PACK64(state[9],  shuffle[2], shuffle[3]);
			PACK64(state[10], shuffle[4], shuffle[5]);
			PACK64(state[11], shuffle[6], shuffle[7]);
		}
		
	}

	// keccak_256(keccak_512(header..nonce) .. mix);
	state[12] = 0x0000000000000001ULL;
	for (uint32_t i = 13; i < 25; i++)
	{
		state[i] = 0ULL;
	}
	state[16] = 0x8000000000000000;
	keccak_f1600_block((uint2 *)state, 1);

	return state[0];
}
#endif

__device__ hash32_t compute_hash(
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t nonce
	)
{
	extern __shared__  compute_hash_share share[];

	// Compute one init hash per work item.
	hash64_t init = init_hash(g_header, nonce);

	// Threads work together in this phase in groups of 8.
	uint32_t const thread_id = threadIdx.x & (THREADS_PER_HASH-1);
	uint32_t const hash_id   = threadIdx.x >> 3;

	hash32_t mix;

	for (int i = 0; i < THREADS_PER_HASH; i++)
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;
		
		uint4 thread_init = share[hash_id].init.uint4s[thread_id & 3];
		
		uint32_t thread_mix = inner_loop(thread_init, thread_id, share[hash_id].mix.uint32s, g_dag);

		share[hash_id].mix.uint32s[thread_id] = thread_mix;
		

		if (i == thread_id)
			mix = share[hash_id].mix;
	}

	return final_hash(&init, &mix);
}

__global__ void 
__launch_bounds__(128, 7)
ethash_search(
	uint32_t* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t start_nonce,
	uint64_t target
	)
{
	
	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;	
	
#if __CUDA_ARCH__ >= SHUFFLE_MIN_VER
	uint64_t hash = compute_hash_shuffle(g_header, g_dag, start_nonce + gid);
	if (cuda_swab64(hash) < target)
	{
		atomicInc(g_output, d_max_outputs);
		g_output[g_output[0]] = gid;
	}
#else
	hash32_t hash = compute_hash(g_header, g_dag, start_nonce + gid);	
	if (cuda_swab64(hash.uint64s[0]) < target)
	{
		atomicInc(g_output,d_max_outputs);
		g_output[g_output[0]] = gid;
	}
#endif
	
	
}

void run_ethash_hash(
	hash32_t* g_hashes,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t start_nonce
) 
{
}

void run_ethash_search(
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream,
	uint32_t* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t start_nonce,
	uint64_t target
)
{
#if __CUDA_ARCH__ >= SHUFFLE_MIN_VER
	ethash_search <<<blocks, threads, 0, stream >>>(g_output, g_header, g_dag, start_nonce, target);
#else
	ethash_search <<<blocks, threads, (sizeof(compute_hash_share) * threads) / THREADS_PER_HASH, stream>>>(g_output, g_header, g_dag, start_nonce, target);
#endif
}

cudaError set_constants(
	uint32_t * dag_size,
	uint32_t * max_outputs
	)
{
	cudaError result;
	result = cudaMemcpyToSymbol(d_dag_size, dag_size, sizeof(uint32_t));
	result = cudaMemcpyToSymbol(d_max_outputs, max_outputs, sizeof(uint32_t));
	return result;
}
