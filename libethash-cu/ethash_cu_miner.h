#pragma once

#define __CL_ENABLE_EXCEPTIONS 
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "cl.hpp"
#pragma clang diagnostic pop
#else
#include <cuda_runtime.h>
#endif

#include <time.h>
#include <functional>
#include <libethash/ethash.h>
#include "ethash_cu_miner_kernel.h"

class ethash_cu_miner
{
public:
	struct search_hook
	{
		virtual ~search_hook(); // always a virtual destructor for a class with virtuals.

		// reports progress, return true to abort
		virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
		virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
	};

public:
	ethash_cu_miner();

	bool init(uint8_t const* _dag, uint64_t _dagSize, unsigned workgroup_size = 64, unsigned _deviceId = 0);
	static std::string platform_info(unsigned _deviceId = 0);
	static int get_num_devices();


	void finish();
	void hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count);
	void search(uint8_t const* header, uint64_t target, search_hook& hook);

private:
	enum { c_max_search_results = 63, c_num_buffers = 2, c_hash_batch_size = 1024, c_search_batch_size = 1024*256 };
	
	hash128_t * m_dag_ptr;
	hash32_t * m_header;

	void * m_hash_buf[c_num_buffers];
	uint * m_search_buf[c_num_buffers];
	cudaStream_t  m_streams[c_num_buffers];

	unsigned m_workgroup_size;
};