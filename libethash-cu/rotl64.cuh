#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef USE_ROT_ASM_OPT
#define USE_ROT_ASM_OPT 3
#endif

// 64-bit ROTATE LEFT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
__device__ __forceinline__
uint64_t ROTL64(const uint64_t value, const int offset) {
	uint2 result;
	if (offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	}
	else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shl.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shr.b64 %0, %1, roff;\n\t"
		"add.u64 %0, lhs, %0;\n\t"
		"}\n"
		: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#elif __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 3
__device__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t res;
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
#else
/* host */
#define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))
#endif