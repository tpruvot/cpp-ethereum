/*
* Genoil's CUDA mining kernel for Ethereum
* based on Tim Hughes' opencl kernel.
* thanks to trpuvot,djm34,sp,cbuchner for things i took from ccminer. 
*/

#include "ethash_cu_miner_kernel.h"
#include "ethash_cu_miner_kernel_globals.h"
//#include "rotl64.cuh"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "vector_types.h"

#define GROUP_SIZE 64
#define ACCESSES 64
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
/*
__device__ ulong ROTL64(const ulong x, const int offset)
{
	ulong res;
	asm("{\n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		".reg .pred p;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"setp.lt.u32 p, %2, 32;\n\t"
		"@!p mov.b64 %0, {vl,vh};\n\t"
		"@p  mov.b64 %0, {vh,vl};\n\t"
		"}"
		: "=l"(res) : "l"(x), "r"(offset)
		);
	return res;
}
*/

__device__ ulong ROTL64H(const ulong x, const int offset)
{
	ulong res;
	asm("{\n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"mov.b64 %0, {vl,vh};\n\t"
		"}"
		: "=l"(res) : "l"(x), "r"(offset)
		);
	return res;
}

__device__ ulong ROTL64L(const ulong x, const int offset)
{
	ulong res;
	asm("{\n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"mov.b64 %0, {vh,vl};\n\t"
		"}"
		: "=l"(res) : "l"(x), "r"(offset)
		);
	return res;
}
__device__ static void keccak_f1600_block(ulong* s, uint out_size)//, uint in_size, uint out_size)
{
	ulong t[5], u, v;
	
	for (size_t i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		u = t[4] ^ ROTL64L(t[1], 1);
		s[0] ^= u; s[5] ^= u; s[10] ^= u; s[15] ^= u; s[20] ^= u;
		u = t[0] ^ ROTL64L(t[2], 1);
		s[1] ^= u; s[6] ^= u; s[11] ^= u; s[16] ^= u; s[21] ^= u;
		u = t[1] ^ ROTL64L(t[3], 1);
		s[2] ^= u; s[7] ^= u; s[12] ^= u; s[17] ^= u; s[22] ^= u;
		u = t[2] ^ ROTL64L(t[4], 1);
		s[3] ^= u; s[8] ^= u; s[13] ^= u; s[18] ^= u; s[23] ^= u;
		u = t[3] ^ ROTL64L(t[0], 1);
		s[4] ^= u; s[9] ^= u; s[14] ^= u; s[19] ^= u; s[24] ^= u;
		 
		/* rho pi: b[..] = rotl(a[..], ..) */
		u = s[1];

		s[1] = ROTL64H(s[6], 44);
		s[6] = ROTL64L(s[9], 20);
		s[9] = ROTL64H(s[22], 61);
		s[22] = ROTL64H(s[14], 39);
		s[14] = ROTL64L(s[20], 18);
		s[20] = ROTL64H(s[2], 62);
		s[2] = ROTL64H(s[12], 43);
		s[12] = ROTL64L(s[13], 25);
		s[13] = ROTL64L(s[19], 8);
		s[19] = ROTL64H(s[23], 56);
		s[23] = ROTL64H(s[15], 41);
		s[15] = ROTL64L(s[4], 27);
		s[4] = ROTL64L(s[24], 14);
		s[24] = ROTL64L(s[21], 2);
		s[21] = ROTL64H(s[8], 55);
		s[8] = ROTL64H(s[16], 45);
		s[16] = ROTL64H(s[5], 36);
		s[5] = ROTL64L(s[3], 28);
		s[3] = ROTL64L(s[18], 21);
		s[18] = ROTL64L(s[17], 15);
		s[17] = ROTL64L(s[11], 10);
		s[11] = ROTL64L(s[7], 6);
		s[7] = ROTL64L(s[10], 3);
		s[10] = ROTL64L(u, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		u = s[0]; v = s[1]; s[0] ^= (~v) & s[2]; 
		
		// squeeze this in here
		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[i];

		// continue chi
		s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & u; s[4] ^= (~u) & v;
		if (i == 23 && out_size == 4) return;
		u = s[5]; v = s[6]; s[5] ^= (~v) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; 
		if (i == 23 && out_size == 8) return;
		s[8] ^= (~s[9]) & u; s[9] ^= (~u) & v;
		u = s[10]; v = s[11]; s[10] ^= (~v) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & u; s[14] ^= (~u) & v;
		u = s[15]; v = s[16]; s[15] ^= (~v) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & u; s[19] ^= (~u) & v;
		u = s[20]; v = s[21]; s[20] ^= (~v) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & u; s[24] ^= (~u) & v;
	}
}

#define copy(dst, src, count) for (uint i = 0; i < count; i++) { (dst)[i] = (src)[i]; }

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

__device__ uint fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

__device__ hash64_t init_hash(hash32_t const* header, ulong nonce)
{
	hash64_t init;
	//uint const init_size = countof(init.ulongs);	//	8
	//uint const hash_size = countof(header->ulongs);	//	4

	// sha3_512(header .. nonce)
	ulong state[25];

	copy(state, header->ulongs, 4);
	state[4] = nonce;
	state[5] = 0x0000000000000001;
	state[6] = 0;
	state[7] = 0;
	state[8] = 0x8000000000000000;
	for (uint i = 9; i < 25; i++)
	{
		state[i] = 0;
	}
	
	keccak_f1600_block(state, 8);// , hash_size + 1, init_size);
	 
	copy(init.ulongs, state, 8);
	return init;
}

__device__ uint inner_loop(uint4 mix, uint thread_id, uint* share, hash128_t const* g_dag)
{
	// share init0
	if (thread_id == 0)
		*share = mix.x;
	//__syncthreads();
	uint init0 = *share;
	
	uint a = 0;

	do
	{
		
		bool update_share = thread_id == ((a >> 2) & (THREADS_PER_HASH-1));

		//#pragma unroll 4
		for (uint i = 0; i < 4; i++)
		{
			
			if (update_share)
			{
				uint m[4] = { mix.x, mix.y, mix.z, mix.w };
				*share = fnv(init0 ^ (a + i), m[i]) % d_dag_size;
			}
			//__syncthreads();
			__threadfence_block();

			mix = fnv4(mix, g_dag[*share].uint4s[thread_id]);
		}
		
	} while ((a += 4) != ACCESSES);// d_acceses);
	
	return fnv_reduce(mix);
}

__device__ hash32_t final_hash(hash64_t const* init, hash32_t const* mix)
{
	ulong state[25];

	hash32_t hash;
	//uint const hash_size = countof(hash.ulongs);	//	4
	//uint const init_size = countof(init->ulongs);	//	8
	//uint const mix_size  = countof(mix->ulongs);	//	4

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->ulongs, 8);
	copy(state + 8, mix->ulongs, 4);
	state[12] = 0x0000000000000001;
	for (uint i = 13; i < 16; i++)
	{
		state[i] = 0;
	}
	state[16] = 0x8000000000000000;
	for (uint i = 17; i < 25; i++)
	{
		state[i] = 0;
	}

	keccak_f1600_block(state,4);// , init_size + mix_size, hash_size);

	// copy out
	copy(hash.ulongs, state, 4);
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
	uint const thread_id = gid & (THREADS_PER_HASH-1);
	uint const hash_id = (gid & (GROUP_SIZE - 1)) >> 3;/// THREADS_PER_HASH;

	hash32_t mix;
	uint i = 0;
	
	do
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;

		uint4 thread_init = share[hash_id].init.uint4s[(thread_id & 3)];

		uint thread_mix = inner_loop(thread_init, thread_id, share[hash_id].mix.uints, g_dag);

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
	__shared__ compute_hash_share share[GROUP_SIZE / THREADS_PER_HASH];
	
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
	//result = cudaMemcpyToSymbol(d_acceses, acceses, sizeof(uint));
	result = cudaMemcpyToSymbol(d_max_outputs, max_outputs, sizeof(uint));
	//result = cudaMemcpyToSymbol(d_workgroup_size, workgroup_size, sizeof(uint));
	return result;
}
