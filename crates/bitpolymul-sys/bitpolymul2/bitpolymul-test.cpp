
#include <stdio.h>
#include <stdlib.h>

#include "benchmark.h"
#include "byte_inline_func.h"

#include "bitpolymul.h"

#include "config_profile.h"

#define TEST_RUN 100
#define REF_RUN TEST_RUN

#define TEST_CONSISTENCY

//#define _HAVE_GF2X_

#ifdef _HAVE_GF2X_
#include "gf2x.h"
void polymul_gf2x( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned terms )
{
	gf2x_mul( c , a , terms , b , terms );
}
#endif


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

//#define bm_func1 bitpolymul_simple
#define bm_func1 bitpolymul_2_64
//#define bm_func1 bitpolymul_2
//#define bm_func1 bitpolymul_128
//#define bm_func1 bitpolymul
#define n_fn1 "fn:" TOSTRING(bm_func1) "()"

#ifdef _HAVE_GF2X_
#define bm_func2 polymul_gf2x
#else
//#define bm_func2 bitpolymul_simple
#define bm_func2 bitpolymul_2_128
#endif
#define n_fn2 "fn:" TOSTRING(bm_func2) "()"


//#define _EXIT_WHILE_FAIL_

#define LEN (1<<16)

#define _DYNA_ALLOC_


#ifdef _PROFILE_
extern "C" {
extern struct benchmark bm_ch;
extern struct benchmark bm_tr;
extern struct benchmark bm_tr2;

extern struct benchmark bm_bc;
extern struct benchmark bm_butterfly;
extern struct benchmark bm_pointmul;
extern struct benchmark bm_pointmul_tower;

extern struct benchmark bm_ich;
extern struct benchmark bm_ibc;
extern struct benchmark bm_ibutterfly;

//extern struct benchmark bm_bm;
//extern struct benchmark bm_mul;
}
#endif

#define LOG2(X) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((X)) - 1))

int main( int argc , char ** argv )
{
	//unsigned char seed[32] = {0};

	unsigned log2_len = LOG2(LEN);
	if( log2_len == LOG2(LEN-1) ) log2_len++;
	if( 2 == argc ) {
		log2_len = atoi( argv[1] );
		if( log2_len > 30 || 0 == log2_len ) {
			printf("Benchmark binary polynomial multiplications.\nUsage: exe [log2_len]\n\n");
			exit(-1);
		}
	}
	unsigned len = 1<<log2_len;

	printf("Multiplication test:\ninput poly len: 2^%d x 64 bit. Benchmark in micro seconds.\n", log2_len );

#ifdef _DYNA_ALLOC_
        uint64_t * poly1 = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*len );
        if( NULL == poly1 ) { printf("alloc 1 fail.\n"); exit(-1); }
        uint64_t * poly2 = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*len );
        if( NULL == poly2 ) { printf("alloc 2 fail.\n"); exit(-1); }
        uint64_t * poly3 = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*len );
        if( NULL == poly3 ) { printf("alloc 3 fail.\n"); exit(-1); }
        uint64_t * poly4 = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*len );
        if( NULL == poly4 ) { printf("alloc 4 fail.\n"); exit(-1); }
        uint64_t * poly5 = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*len );
        if( NULL == poly5 ) { printf("alloc 5 fail.\n"); exit(-1); }
#else
	uint64_t poly1[LEN] __attribute__((aligned(32)));
	uint64_t poly2[LEN] __attribute__((aligned(32)));
	uint64_t poly3[LEN*2] __attribute__((aligned(32)));
	uint64_t poly4[LEN*2] __attribute__((aligned(32)));
#endif

	benchmark bm1;
	bm_init(&bm1);
	benchmark bm2;
	bm_init(&bm2);

#ifdef _PROFILE_
bm_init(&bm_ch);
bm_init(&bm_tr);
bm_init(&bm_tr2);
//bm_init(&bm_bm);
//bm_init(&bm_mul);
bm_init(&bm_bc);
bm_init(&bm_butterfly);
bm_init(&bm_pointmul);
bm_init(&bm_pointmul_tower);
bm_init(&bm_ibc);
bm_init(&bm_ibutterfly);
bm_init(&bm_ich);
#endif

	//byte_rand( poly1 , LEN );
//	memset( poly1 , 0 , sizeof(uint64_t)*len );
//	for(unsigned i=0;i<len;i++) poly1[i] = i;
//	poly1[0] = 3;
//	poly1[1] = 3;
//	memset( poly2 , 0 , sizeof(uint64_t)*len );
//	for(unsigned i=0;i<len;i++) poly2[i] = i+1;
//	poly2[0] = 2;
//	poly2[1] = 2;
	//memcpy( poly2 , poly1 , LEN*sizeof(uint64_t) );
	for(unsigned q=0;q<len;q++) { poly1[q] = rand(); poly1[q]<<=32; poly1[q] |= rand(); }
	for(unsigned q=0;q<len;q++) { poly2[q] = rand(); poly2[q]<<=32; poly2[q] |= rand(); }
	bm_func1( poly5 , poly2 , poly1 , len );
	bm_func1( poly3 , poly1 , poly2 , len );
	if( 32 >= len ) {
	printf("poly1 :" ); u64_dump( poly1 , len ); puts("");
	printf("poly2 :" ); u64_dump( poly2 , len ); puts("");
	printf("poly3 :" ); u64_dump( poly3 , len*2 ); puts("");
	byte_xor( poly5 , poly3 , len*2 );
		if( ! byte_is_zero( poly5 , len*2 ) ) {
			printf("consistency fail: \n");
			printf("diff:"); u64_fdump(stdout,poly5,len*2); puts("");
		}
	}

	for(unsigned q=0;q<len;q++) { poly1[q] = rand(); poly1[q]<<=32; poly1[q] |= rand(); }
	for(unsigned q=0;q<len;q++) { poly2[q] = rand(); poly2[q]<<=32; poly2[q] |= rand(); }
	bm_func1( poly3 , poly1 , poly2 , len );
	//bm_func1( poly4 , poly1 , poly2 , len );
	bm_func1( poly4 , poly2 , poly1 , len );
	if( 32 >= len ) {
	printf("poly1 :" ); u64_dump( poly1 , len ); puts("");
	printf("poly2 :" ); u64_dump( poly2 , len ); puts("");
	printf("poly3 :" ); u64_dump( poly3 , len*2 ); puts("");
	byte_xor( poly4 , poly3 , len*2 );
		if( ! byte_is_zero( poly4 , len*2 ) ) {
			printf("consistency fail: \n");
			printf("diff:"); u64_fdump(stdout,poly4,len*2); puts("");
		}
	}

	//for(int i=0;i<7;i++) bm_func2( o1 , poly1 , v1 );
	//exit(-1);

	unsigned fail_count = 0;
	unsigned chk = 0;
	for(unsigned i=0;i<TEST_RUN;i++) {
		//byte_rand( poly1 , LEN );
		//memcpy( poly2 , poly1 , LEN*sizeof(uint64_t) );
		//srand(0);
		for(unsigned q=0;q<len;q++) { poly1[q] = rand(); poly1[q]<<=32; poly1[q] |= rand(); }
		for(unsigned q=0;q<len;q++) { poly2[q] = rand(); poly2[q]<<=32; poly2[q] |= rand(); }
		for(unsigned q=0;q<len;q++) { poly1[q] = q+1; }
		for(unsigned q=0;q<len;q++) { poly2[q] = q+2; }

BENCHMARK( bm1 , {
		bm_func1( poly3 , poly2 , poly1 , len );
} );
		if(i >= REF_RUN) continue;
BENCHMARK( bm2 , {
		bm_func2( poly4 , poly2 , poly1 , len );
} );

		memcpy( poly5 , poly4 , sizeof(uint64_t)*len*2 );
		byte_xor( poly5 , poly3 , len*2 );
		chk |= ((unsigned*)(&poly5[0]))[0];
		if( ! byte_is_zero( poly5 , len*2 ) ) {
			fail_count ++;
#ifdef TEST_CONSISTENCY
			printf("consistency fail: %d.\n", i);
			printf("res1:"); u64_fdump(stdout,poly3,len*2); puts("");
			printf("res2:"); u64_fdump(stdout,poly4,len*2); puts("");
			printf("diff:"); u64_fdump(stdout,poly5,len*2); puts("");
			printf("\n");
			exit(-1);
#endif
		}
	}

	printf("fail count: %d.\n", fail_count );
	printf("check: %x\n", chk );
	char msg[256];
	bm_dump( msg , 256 , &bm1 );
	printf("benchmark (%s) :\n%s\n", n_fn1 , msg );

	bm_dump( msg , 256 , &bm2 );
	printf("benchmark (%s) :\n%s\n\n", n_fn2 , msg );

#ifdef _PROFILE_
	bm_dump( msg , 256 , &bm_tr ); printf("benchmark (tr ) :\n%s\n", msg );
	bm_dump( msg , 256 , &bm_tr2 ); printf("benchmark (tr2) :\n%s\n", msg );
//	bm_dump( msg , 256 , &bm_mul ); printf("benchmark (mul) :\n%s\n", msg );
//	bm_dump( msg , 256 , &bm_bm ); printf("benchmark (bm ) :\n%s\n\n", msg );

	bm_dump( msg , 256 , &bm_ch ); printf("benchmark (ch ) :\n%s\n", msg );
	bm_dump( msg , 256 , &bm_bc ); printf("benchmark (bc) :\n%s\n", msg );
	bm_dump( msg , 256 , &bm_butterfly ); printf("benchmark (butterfly) :\n%s\n", msg );
	bm_dump( msg , 256 , &bm_pointmul ); printf("benchmark (pointmul) :\n%s\n", msg );
	bm_dump( msg , 256 , &bm_pointmul_tower ); printf("benchmark (pointmul_tower) :\n%s\n", msg );
	bm_dump( msg , 256 , &bm_ibutterfly ); printf("benchmark (ibutterfly) :\n%s\n", msg );
	bm_dump( msg , 256 , &bm_ibc ); printf("benchmark (ibc) :\n%s\n", msg );
	bm_dump( msg , 256 , &bm_ich ); printf("benchmark (ich ) :\n%s\n\n", msg );
#endif

#ifdef _DYNA_ALLOC_
        free( poly1 );
        free( poly2 );
        free( poly3 );
        free( poly4 );
#endif

	return 0;
}
