/*
* Genoil's CUDA mining kernel for Ethereum
* based on Tim Hughes' opencl kernel.
* thanks to trpuvot,djm34,sp,cbuchner for things i took from ccminer. 
*/

#include "ethash_cu_miner_kernel.h"
#include "ethash_cu_miner_kernel_globals.h"
#include "rotl64.cuh"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "vector_types.h"

#define THREADS_PER_HASH (128 / 16)
#define HASHES_PER_LOOP (GROUP_SIZE / THREADS_PER_HASH)

#define FNV_PRIME	0x01000193

__device__ __constant__ ulong const keccak_round_constants[24] = {
	0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
	0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
	0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
	0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
	0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

//#define ROTL64(a, offset) ((a) << (offset)) ^ ((a) >> (64 - offset))

__device__ static void keccak_f1600_no_absorb(ulong* s, uint in_size, uint out_size)
{
	for (uint i = in_size+1; i < 25; i++)
	{
		s[i] = 0;
	}

	s[in_size] = 0x0000000000000001; // was ^=, but was always 0 anyway
	s[24 - out_size * 2] = 0x8000000000000000; // was ^=. but was always 0 anyway

	ulong t[5], u[5], v, w;
	
	for (size_t i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];
		/*
		u = t[4] ^ ROTL64(t[1], 1);
		s[0] ^= u; s[5] ^= u; s[10] ^= u; s[15] ^= u; s[20] ^= u;
		u = t[0] ^ ROTL64(t[2], 1);
		s[1] ^= u; s[6] ^= u; s[11] ^= u; s[16] ^= u; s[21] ^= u;
		u = t[1] ^ ROTL64(t[3], 1);
		s[2] ^= u; s[7] ^= u; s[12] ^= u; s[17] ^= u; s[22] ^= u;
		u = t[2] ^ ROTL64(t[4], 1);
		s[3] ^= u; s[8] ^= u; s[13] ^= u; s[18] ^= u; s[23] ^= u;
		u = t[3] ^ ROTL64(t[0], 1);
		s[4] ^= u; s[9] ^= u; s[14] ^= u; s[19] ^= u; s[24] ^= u;
		*/
		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROTL64(s[6], 44);
		s[6] = ROTL64(s[9], 20);
		s[9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[2], 62);
		s[2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19], 8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[4], 27);
		s[4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21], 2);
		s[21] = ROTL64(s[8], 55);
		s[8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[5], 36);
		s[5] = ROTL64(s[3], 28);
		s[3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[7], 6);
		s[7] = ROTL64(s[10], 3);
		s[10] = ROTL64(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[i];
	}
}

#define copy(dst, src, count) for (uint i = 0; i < count; i++) { (dst)[i] = (src)[i]; }

#define countof(x) (sizeof(x) / sizeof(x[0]))

#define fnv(x,y) ((x) * FNV_PRIME ^(y))

__device__  void fnv4(uint * x, const uint * y)
{
	x[0] = fnv(x[0], y[0]);
	x[1] = fnv(x[1], y[1]);
	x[2] = fnv(x[2], y[2]);
	x[3] = fnv(x[3], y[3]);
}


__device__ uint  fnv_reduce(uint * v)
{
	return fnv(fnv(fnv(v[0], v[1]), v[2]), v[3]);
}

__device__ hash64_t init_hash(hash32_t const* header, ulong nonce)
{
	hash64_t init;
	uint const init_size = countof(init.ulongs);
	uint const hash_size = countof(header->ulongs);

	// sha3_512(header .. nonce)
	ulong state[25];
	copy(state, header->ulongs, hash_size);
	state[hash_size] = nonce;
	keccak_f1600_no_absorb(state, hash_size + 1, init_size);
	 
	copy(init.ulongs, state, init_size);
	return init;
}

__device__ uint inner_loop(uint* mix, uint thread_id, uint* share, hash128_t const* g_dag)
{
	// share init0
	if (thread_id == 0)
		*share = mix[0];
	//__syncthreads();
	uint init0 = *share;
	
	uint a = 0;
	uint t4 = thread_id * 4;

	do
	{
		
		bool update_share = thread_id == (a / 4) % THREADS_PER_HASH;

		#pragma unroll 4
		for (uint i = 0; i < 4; i++)
		{
			
			if (update_share)
			{
				*share = 32 * (fnv(init0 ^ (a + i), mix[i]) % d_dag_size);
			}
			__syncthreads();
			fnv4(mix, &(g_dag[*share+t4]));
		}
		
	} while ((a += 4) != 64);// d_acceses);
	
	return fnv_reduce(mix);
}

__device__ hash32_t final_hash(hash64_t const* init, hash32_t const* mix)
{
	ulong state[25];

	hash32_t hash;
	uint const hash_size = countof(hash.ulongs);
	uint const init_size = countof(init->ulongs);
	uint const mix_size  = countof(mix->ulongs);

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->ulongs, init_size);
	copy(state + init_size, mix->ulongs, mix_size);
	keccak_f1600_no_absorb(state, init_size + mix_size, hash_size);

	// copy out
	copy(hash.ulongs, state, hash_size);
	return hash;
}

typedef union
{
	hash64_t init;
	hash32_t mix;
} compute_hash_share;

__device__ hash32_t compute_hash(
	compute_hash_share* share,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong gid
	)
{
	// Compute one init hash per work item.
	hash64_t init = init_hash(g_header, start_nonce + gid);

	// Threads work together in this phase in groups of 8.
	uint const thread_id = gid % THREADS_PER_HASH;
	uint const hash_id = (gid % d_workgroup_size) / THREADS_PER_HASH;

	hash32_t mix;
	uint i = 0;
	uint t4 = 4 * (thread_id % 4);

	do
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;

		uint * thread_init = &(share[hash_id].init.uints[t4]);

		uint t[4] = { thread_init[0], thread_init[1], thread_init[2], thread_init[3] };
		uint thread_mix = inner_loop(t, thread_id, share[hash_id].mix.uints, g_dag);

		share[hash_id].mix.uints[thread_id] = thread_mix;

		if (i == thread_id)
			mix = share[hash_id].mix;

	} while (++i != THREADS_PER_HASH );

	return final_hash(&init, &mix);

}

__global__ void ethash_search(
	uint* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target
	)
{
	__shared__ compute_hash_share share[64];// = new compute_hash_share[d_workgroup_size / THREADS_PER_HASH];
	
	uint const gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	hash32_t hash = compute_hash(share, g_header, g_dag, start_nonce, gid);
	
	if (__brevll(hash.ulongs[0]) < target)
	{
		atomicInc(g_output,d_max_outputs);
		g_output[g_output[0]] = gid;
	}
	
}

void run_ethash_hash(
	hash32_t* g_hashes,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce
) 
{
}

void run_ethash_search(
	uint blocks,
	uint threads,
	cudaStream_t stream,
	uint* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target
)
{
	ethash_search<<<blocks, threads, 0, stream>>>(g_output, g_header, g_dag, start_nonce, target);
}

cudaError set_constants(
	uint * dag_size,
	uint * acceses,
	uint * max_outputs,
	uint * workgroup_size
	)
{
	cudaError result;
	result = cudaMemcpyToSymbol(d_dag_size, dag_size, sizeof(uint));
	result = cudaMemcpyToSymbol(d_acceses, acceses, sizeof(uint));
	result = cudaMemcpyToSymbol(d_max_outputs, max_outputs, sizeof(uint));
	result = cudaMemcpyToSymbol(d_workgroup_size, workgroup_size, sizeof(uint));
	return result;
}
