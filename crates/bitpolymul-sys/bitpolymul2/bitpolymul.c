/*
Copyright (C) 2017 Ming-Shing Chen

This file is part of BitPolyMul.

BitPolyMul is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BitPolyMul is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with BitPolyMul.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "bc.h"

#include "butterfly_net.h"


#include "config_profile.h"

#define MAX_TERMS 65536

#include "gfext_aesni.h"


#ifdef _PROFILE_

#include "benchmark.h"

struct benchmark bm_ch;
struct benchmark bm_bc;
struct benchmark bm_butterfly;
struct benchmark bm_pointmul;
struct benchmark bm_pointmul_tower;

struct benchmark bm_ich;
struct benchmark bm_ibc;
struct benchmark bm_ibutterfly;

struct benchmark bm_tr;
struct benchmark bm_tr2;

#endif


#define LOG2(X) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((X)) - 1))

// for removing warning.
void *aligned_alloc( size_t alignment, size_t size );


void bitpolymul_simple( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned _n_64 )
{
	if( 0 == _n_64 ) return;
	unsigned n_64 = 0;
	if( 1 == _n_64 ) n_64 = _n_64;
	else {
		unsigned log_2_n64 = LOG2(_n_64);
		unsigned log_2_n64_1 = LOG2(_n_64-1);
		if( log_2_n64 == log_2_n64_1 )log_2_n64 += 1;
		n_64 = 1<<log_2_n64;
	}

	//uint64_t a_bc[MAX_TERMS] __attribute__((aligned(32)));
	//uint64_t b_bc[MAX_TERMS] __attribute__((aligned(32)));
	uint64_t * a_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	if( NULL == a_bc ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	if( NULL == b_bc ) { printf("alloc fail.\n"); exit(-1); }

#ifdef _PROFILE_SIMPLE_
bm_start(&bm_bc);
#endif
	memcpy( a_bc , a , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) a_bc[i] = 0;
	bc_to_lch( a_bc , n_64 );

	memcpy( b_bc , b , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) b_bc[i] = 0;
	bc_to_lch( b_bc , n_64 );
#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_bc);
#endif

	unsigned n_terms = 2*n_64;

	//uint64_t a_fx[4*MAX_TERMS] __attribute__((aligned(32))) = {0};
	//uint64_t b_fx[4*MAX_TERMS] __attribute__((aligned(32))) = {0};
	uint64_t * a_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*n_terms );
	if( NULL == a_fx ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*n_terms );
	if( NULL == b_fx ) { printf("alloc fail.\n"); exit(-1); }

	for(unsigned i=0;i<_n_64;i++) {
		a_fx[2*i] = a_bc[i];
		a_fx[2*i+1] = 0;
	}
	for(unsigned i=_n_64;i<n_64;i++) { a_fx[2*i]=0; a_fx[2*i+1]=0; }
	for(unsigned i=0;i<_n_64;i++) {
		b_fx[2*i] = b_bc[i];
		b_fx[2*i+1] = 0;
	}
	for(unsigned i=_n_64;i<n_64;i++) { b_fx[2*i]=0; b_fx[2*i+1]=0; }

#ifdef _PROFILE_SIMPLE_
bm_start(&bm_butterfly);
#endif
	butterfly_net_half_inp_clmul( a_fx , n_terms );
	butterfly_net_half_inp_clmul( b_fx , n_terms );

#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_butterfly);
bm_start(&bm_pointmul);
#endif
	for(unsigned i=0;i<n_terms;i++) {
		gf2ext128_mul_sse( (uint8_t *)&a_fx[i*2] , (uint8_t *)&a_fx[i*2] , (uint8_t*)& b_fx[i*2] );
	}

#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_pointmul);
bm_start(&bm_ibutterfly);
#endif

	i_butterfly_net_clmul( a_fx , n_terms );

#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_ibutterfly);
bm_start(&bm_ibc);
#endif
	bc_to_mono_128( a_fx , n_terms );

#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_ibc);
#endif


	c[0] = a_fx[0];
	for(unsigned i=1;i<(2*_n_64);i++) {
		c[i] = a_fx[i*2];
		c[i] ^= a_fx[(i-1)*2+1];
	}

	free(a_bc);
	free(b_bc);
	free(a_fx);
	free(b_fx);

}



/////////////////////////////////////////////////////////////////////////////////


#include "btfy.h"
#include "encode.h"

void bitpolymul_2_128( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned _n_64 )
{
	if( 0 == _n_64 ) return;
	unsigned n_64 = 0;
	if( 1 == _n_64 ) n_64 = _n_64;
	else {
		unsigned log_2_n64 = LOG2(_n_64);
		unsigned log_2_n64_1 = LOG2(_n_64-1);
		if( log_2_n64 == log_2_n64_1 )log_2_n64 += 1;
		n_64 = 1<<log_2_n64;
	}

	if( 256 > n_64 ) n_64 = 256;

	uint64_t * a_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
		//uint64_t * a_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64*2 );
	if( NULL == a_bc ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
		//uint64_t * b_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64*2 );
	if( NULL == b_bc ) { printf("alloc fail.\n"); exit(-1); }

#ifdef _PROFILE_
bm_start(&bm_bc);
#endif
	memcpy( a_bc , a , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) a_bc[i] = 0;
	//bc_to_lch_2( a_bc , n_64 );
	bc_to_lch_2_unit256( a_bc , n_64 );
		//for(unsigned i=n_64;i<n_64*2;i++) a_bc[i] = 0;

	memcpy( b_bc , b , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) b_bc[i] = 0;
	//bc_to_lch_2( b_bc , n_64 );
	bc_to_lch_2_unit256( b_bc , n_64 );
		//for(unsigned i=n_64;i<n_64*2;i++) b_bc[i] = 0;

#ifdef _PROFILE_
bm_stop(&bm_bc);
#endif

	unsigned n_terms = n_64;
	unsigned log_n = __builtin_ctz( n_terms );
	uint64_t * a_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*n_terms );
	if( NULL == a_fx ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*n_terms );
	if( NULL == b_fx ) { printf("alloc fail.\n"); exit(-1); }

#ifdef _PROFILE_
bm_start(&bm_tr);
#endif
	encode_128_half_input_zero( a_fx , a_bc , n_terms );
		//encode( a_fx , a_bc , n_terms );

	encode_128_half_input_zero( b_fx , b_bc , n_terms );
		//encode( b_fx , b_bc , n_terms );

#ifdef _PROFILE_
bm_stop(&bm_tr);
#endif
#ifdef _PROFILE_
bm_start(&bm_butterfly);
#endif
	btfy_128( b_fx , n_terms , 64+log_n+1 );
	btfy_128( a_fx , n_terms , 64+log_n+1 );
#ifdef _PROFILE_
bm_stop(&bm_butterfly);
#endif


#ifdef _PROFILE_
bm_start(&bm_pointmul);
#endif
	for(unsigned i=0;i<n_terms;i++) gf2ext128_mul_sse( (uint8_t *)&a_fx[i*2] , (uint8_t *)&a_fx[i*2] , (uint8_t*)& b_fx[i*2] );
#ifdef _PROFILE_
bm_stop(&bm_pointmul);
#endif

#ifdef _PROFILE_
bm_start(&bm_ibutterfly);
#endif
	i_btfy_128( a_fx , n_terms , 64+log_n+1 );
#ifdef _PROFILE_
bm_stop(&bm_ibutterfly);
#endif

#ifdef _PROFILE_
bm_start(&bm_tr2);
#endif
	decode_128( b_fx , a_fx , n_terms );
#ifdef _PROFILE_
bm_stop(&bm_tr2);
#endif

#ifdef _PROFILE_
bm_start(&bm_ibc);
#endif
	//bc_to_mono_2( b_fx , 2*n_64 );
	bc_to_mono_2_unit256( b_fx , 2*n_64 );
#ifdef _PROFILE_
bm_stop(&bm_ibc);
#endif

	for(unsigned i=0;i<(2*_n_64);i++) {
		c[i] = b_fx[i];
	}

	free(a_bc);
	free(b_bc);
	free(a_fx);
	free(b_fx);

}






///////////////////////////////////////////////////


void bitpolymul_2_64( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned _n_64 )
{
	if( 0 == _n_64 ) return;
	if( _n_64 > (1<<26) ) { printf("un-supported length of polynomials."); exit(-1); }
	unsigned n_64 = 0;
	if( 1 == _n_64 ) n_64 = _n_64;
	else {
		unsigned log_2_n64 = LOG2(_n_64);
		unsigned log_2_n64_1 = LOG2(_n_64-1);
		if( log_2_n64 == log_2_n64_1 )log_2_n64 += 1;
		n_64 = 1<<log_2_n64;
	}

	if( 256 > n_64 ) n_64 = 256;

	uint64_t * a_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	//uint64_t * a_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64*2 );
	if( NULL == a_bc ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	//uint64_t * b_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64*2 );
	if( NULL == b_bc ) { printf("alloc fail.\n"); exit(-1); }

#ifdef _PROFILE_
bm_start(&bm_bc);
#endif
	memcpy( a_bc , a , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) a_bc[i] = 0;
	//for(unsigned i=n_64;i<2*n_64;i++) a_bc[i] = 0;
	//bc_to_lch_2( a_bc , n_64 );
	bc_to_lch_2_unit256( a_bc , n_64 );
		//for(unsigned i=n_64;i<n_64*2;i++) a_bc[i] = 0;

	memcpy( b_bc , b , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) b_bc[i] = 0;
	//for(unsigned i=n_64;i<2*n_64;i++) b_bc[i] = 0;
	//bc_to_lch_2( b_bc , n_64 );
	bc_to_lch_2_unit256( b_bc , n_64 );
		//for(unsigned i=n_64;i<n_64*2;i++) b_bc[i] = 0;

#ifdef _PROFILE_
bm_stop(&bm_bc);
#endif

	unsigned n_terms = n_64*2;
	unsigned log_n = __builtin_ctz( n_terms );
	uint64_t * a_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_terms );
	if( NULL == a_fx ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_terms );
	if( NULL == b_fx ) { printf("alloc fail.\n"); exit(-1); }

#ifdef _PROFILE_
bm_start(&bm_tr);
#endif
	encode_64_half_input_zero( a_fx , a_bc , n_terms );
	encode_64_half_input_zero( b_fx , b_bc , n_terms );

#ifdef _PROFILE_
bm_stop(&bm_tr);
#endif
#ifdef _PROFILE_
bm_start(&bm_butterfly);
#endif
	btfy_64( b_fx , n_terms , 32+log_n+1 );
	btfy_64( a_fx , n_terms , 32+log_n+1 );
#ifdef _PROFILE_
bm_stop(&bm_butterfly);
#endif


#ifdef _PROFILE_
bm_start(&bm_pointmul);
#endif
	for(unsigned i=0;i<n_terms;i+=4) {
		_mm_prefetch( &a_fx[i+4] , _MM_HINT_T0 );
		_mm_prefetch( &b_fx[i+4] , _MM_HINT_T0 );
		gf2ext64_mul_4x4_avx2( (uint8_t *)&a_fx[i] , (uint8_t *)&a_fx[i] , (uint8_t*)& b_fx[i] );
	}
#ifdef _PROFILE_
bm_stop(&bm_pointmul);
#endif

#ifdef _PROFILE_
bm_start(&bm_ibutterfly);
#endif
	i_btfy_64( a_fx , n_terms , 32+log_n+1 );
#ifdef _PROFILE_
bm_stop(&bm_ibutterfly);
#endif

#ifdef _PROFILE_
bm_start(&bm_tr2);
#endif
	decode_64( b_fx , a_fx , n_terms );
#ifdef _PROFILE_
bm_stop(&bm_tr2);
#endif

#ifdef _PROFILE_
bm_start(&bm_ibc);
#endif
	bc_to_mono_2_unit256( b_fx , n_terms );
#ifdef _PROFILE_
bm_stop(&bm_ibc);
#endif

	for(unsigned i=0;i<(2*_n_64);i++) {
		c[i] = b_fx[i];
	}

	free(a_bc);
	free(b_bc);
	free(a_fx);
	free(b_fx);

}






