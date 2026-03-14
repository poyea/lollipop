/*
 * _mc_rng.cuh -- Lightweight per-thread RNG for Monte Carlo kernels.
 *
 * Uses xorshift32: three shifts and three XORs per step.  It is the
 * smallest RNG that still passes basic statistical tests, needs no
 * shared memory or library calls, and compiles to a handful of ALU
 * instructions -- ideal for GPU Monte Carlo where throughput matters
 * more than cryptographic quality.
 */

#ifndef _MC_RNG_CUH
#define _MC_RNG_CUH

#include <cstdint>

/*
 * mc_seed -- derive a per-thread seed from a user-supplied base value.
 *
 * Combines the linear thread index with `base` through a
 * multiply-xorshift mix (splitmix-style) so that neighbouring threads
 * start from uncorrelated states even when `base` is a small integer.
 */
__device__ __forceinline__
uint32_t mc_seed(uint32_t base) {
    uint32_t s = (blockIdx.x * blockDim.x + threadIdx.x) ^ base;
    s ^= s >> 16;
    s *= 0x45d9f3bU;
    s ^= s >> 16;
    s *= 0x45d9f3bU;
    s ^= s >> 16;
    /* Ensure state is never zero (xorshift32 fixed point). */
    return s | 1U;
}

/*
 * mc_next -- one xorshift32 step.  Mutates *state and returns the new
 * value.  Period is 2^32 - 1.
 */
__device__ __forceinline__
uint32_t mc_next(uint32_t* state) {
    uint32_t s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    return s;
}

/*
 * mc_float -- uniform float in [0, 1).
 *
 * Takes the upper 24 bits of the next random value and divides by
 * 2^24 so the result is exactly representable in IEEE-754 float32.
 */
__device__ __forceinline__
float mc_float(uint32_t* s) {
    return (mc_next(s) >> 8) * (1.0f / 16777216.0f);  /* 1 / 2^24 */
}

/*
 * mc_int -- uniform integer in [0, n).
 *
 * Uses a simple modulo reduction.  Bias is negligible for n << 2^32,
 * which is the common case in Monte Carlo simulations.
 */
__device__ __forceinline__
int mc_int(uint32_t* s, int n) {
    return (int)(mc_next(s) % (uint32_t)n);
}

#endif /* _MC_RNG_CUH */
