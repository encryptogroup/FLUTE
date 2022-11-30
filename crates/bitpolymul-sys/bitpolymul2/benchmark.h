/*
 * $Id: benchmark.h 1271 2008-06-08 08:06:13Z owenhsin $
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

/* #define CONFIG_BENCH_SYSTIME */

#define CONFIG_BENCH_SYSTIME

#if defined(CONFIG_BENCH_SYSTIME)
#include <sys/time.h>
#endif

/* Copied from http://en.wikipedia.org/wiki/RDTSC */
#ifdef __cplusplus
extern "C"
{
#endif
#if !defined(CONFIG_BENCH_SYSTIME)
#if defined(CONFIG_NEON)
static inline uint32_t
rdtsc32(void)
{
#if defined(__GNUC__) && defined(__ARM_ARCH_7A__)
	uint32_t r = 0;
	asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(r) );
	return r;
#else
#error Unsupported architecture/compiler!
#endif
}
#else
static inline uint64_t rdtsc() {
	uint32_t lo, hi;
	/* We cannot use "=A", since this would use %rax on x86_64 */
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
	return (uint64_t)hi << 32 | lo;
}
#endif
#endif
#ifdef __cplusplus
}
#endif

#define RECMAX 6

#define BENCHMARK(bm,call) do { \
		bm_start(&(bm)); \
		call; \
		bm_stop(&(bm)); \
	} while (0)

struct benchmark {
#if defined(CONFIG_BENCH_SYSTIME)
	struct timeval start;
	struct timeval stop;
#else
	uint64_t start;
	uint64_t stop;
#endif
	double record[RECMAX];
	double acc;
	int currec;
	int count;
};

static inline void
bm_init(struct benchmark *bm)
{
	memset(bm, 0, sizeof(*bm));
}

static inline void
bm_start(struct benchmark *bm)
{
#if defined(CONFIG_BENCH_SYSTIME)
	gettimeofday(& bm->start , NULL);
#else
#if defined(CONFIG_NEON)
	bm->start = rdtsc32();
#else
	bm->start = rdtsc();
#endif
#endif
}

static inline void
bm_stop(struct benchmark *bm)
{
#if defined(CONFIG_BENCH_SYSTIME)
	gettimeofday(& bm->stop , NULL);
	bm->record[bm->currec] = (bm->stop.tv_sec - bm->start.tv_sec)*1000000.0; /* sec to us */
	bm->record[bm->currec] += bm->stop.tv_usec - bm->start.tv_usec;
#else
#if defined(CONFIG_NEON)
	bm->stop = rdtsc32();
#else
	bm->stop = rdtsc();
#endif
	bm->record[bm->currec] = bm->stop - bm->start;
#endif
	bm->acc += bm->record[bm->currec];
	bm->currec = (bm->currec + 1) % RECMAX;
	++bm->count;
}

static inline void
bm_dump(char *buf, size_t bufsize, const struct benchmark *bm)
{
	int i;
	size_t len;
#if defined(CONFIG_BENCH_SYSTIME)
	const char * unit = "micro sec.";
#else
	const char * unit = "cycles";
#endif

	len = snprintf(buf, bufsize, "%.0lf (%s, avg. of %d):", bm->acc/bm->count, unit, bm->count);
	buf += len;
	bufsize -= len;
	for (i = 0; i < RECMAX; ++i) {
		len = snprintf(buf, bufsize, " %.0lf", bm->record[i]);
		buf += len;
		bufsize -= len;
	}
}

#endif /* BENCHMARK_H */
