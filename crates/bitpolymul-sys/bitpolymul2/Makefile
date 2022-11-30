CC=    gcc
CXX=   g++
LD=    gcc

CFLAGS= -O3 -funroll-loops -mavx2 -mpclmul -std=gnu99 -Wextra -Wall
CXXFLAGS= -O3  -mavx2 -mpclmul -fno-exceptions -fno-rtti -nostdinc++ -Wextra -Wall
INCPATH= -I/usr/local/include -I/opt/local/include -I/usr/include #-I../../nist-mq-submission/gf2-dev/
LDFLAGS= 
LIBPATH= -L/usr/local/lib -L/opt/local/lib -L/usr/lib #-L../gf2-dev/
LIBS=    #-lm -lcrypto -lgf2x


OBJ= bc.o bitpolymul.o encode.o butterfly_net.o gf2128_cantor_iso.o btfy.o trunc_btfy_tab.o gf264_cantor_iso.o trunc_btfy_tab_64.o
EXE= bitpolymul-test #bc-test


CSRC= $(wildcard *.cpp)


ifdef DEBUG
        CFLAGS+=  -D_DEBUG_
        CXXFLAGS+= -D_DEBUG_
endif

ifdef NO_SSE
	CFLAGS += -D_NO_SSE_
	CXXFLAGS += -D_NO_SSE_
endif


ifdef K
	CFLAGS += -DK=$(K)
	CXXFLAGS += -DK=$(K)
endif


ifdef AVX2
	CFLAGS += -mavx2 -D_USE_AVX2_
	CXXFLAGS += -mavx2 -D_USE_AVX2_
endif

ifdef AVX
	CFLAGS += -mavx -D_USE_AVX_
	CXXFLAGS += -mavx -D_USE_AVX_
endif

ifdef GPROF
	CFLAGS += -pg
	CXXFLAGS += -pg
	LDFLAGS += -pg
endif

.PHONY: all tests tables clean

all: $(OBJ) $(EXE)


%-test: $(OBJ) %-test.o
	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^ $(LIBS)

%-benchmark: $(OBJ) %-benchmark.o
	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCPATH) -c $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCPATH) -c $<

tests:
	cd tests; make

tables:
	cd supplement; make

clean:
	rm *.o *-test *-benchmark
