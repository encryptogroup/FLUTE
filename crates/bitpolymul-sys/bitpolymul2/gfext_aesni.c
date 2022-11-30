
#include <stdint.h>
#include <immintrin.h>
#include "gfext_aesni.h"

const uint64_t _gf2ext64_reducer[4] __attribute__((aligned(32)))  = {0x415A776C2D361B00ULL,0x1bULL,0x415A776C2D361B00ULL,0x1bULL};

const uint64_t _gf2ext128_reducer[2] __attribute__((aligned(32)))  = {0x87ULL,0x0ULL};

const uint64_t _s7[2] __attribute__((aligned(32)))  = {0x100010116ULL,0x1ULL};

extern __m128i _gf2ext64_reduce_sse( __m128i x0 );
extern __m128i _gf2ext64_reduce_x2_sse( __m128i x0 , __m128i y0 );
extern __m256i _gf2ext64_reduce_x4_avx2( __m128i w0 , __m128i x0 , __m128i y0 , __m128i z0 );
extern __m128i _gf2ext64_mul_sse( __m128i a0 , __m128i b0 );
extern __m128i _gf2ext64_mul_hi_sse( __m128i a0 , __m128i b0 );
extern __m128i _gf2ext64_mul_2x1_sse( __m128i a0a1 , __m128i b0 );
extern __m256i _gf2ext64_mul_4x1_avx2( __m256i a , __m128i b0 );
extern __m128i _gf2ext64_mul_2x2_sse( __m128i a0a1 , __m128i b0b1 );
extern __m256i _gf2ext64_mul_4x4_avx2( __m256i a , __m256i b );
extern uint64_t gf2ext64_mul_u64( uint64_t a , uint64_t b );
extern void gf2ext64_mul_sse( uint8_t * c , const uint8_t * a , const uint8_t * b );
extern void gf2ext64_mul_2x2_sse( uint8_t * c , const uint8_t * a , const uint8_t * b );
extern void gf2ext64_mul_4x4_avx2( uint8_t * c , const uint8_t * a , const uint8_t * b );
extern __m128i _gf2ext128_reduce_sse( __m128i x0 , __m128i x128 );
extern __m128i _gf_aesgcm_reduce_sse( __m128i x0 , __m128i x128 );
extern void gf2ext128_mul_sse( uint8_t * c , const uint8_t * a , const uint8_t * b );
extern __m128i _gf2ext128_mul_sse( __m128i a0 , __m128i b0 );
extern __m256i _gf2ext128_mul_2x1_avx2( __m256i a0a1 , __m128i b0 );
extern __m256i div_s7( __m256i a );
extern __m256i exp_s7( __m256i a );
