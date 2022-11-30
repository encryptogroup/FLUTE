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

#ifndef _BITPOLYMUL_H_
#define _BITPOLYMUL_H_


#include <stdint.h>

#define bitpolymul bitpolymul_2

#ifdef  __cplusplus
extern  "C" {
#endif


void bitpolymul_simple( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned n_64 );


void bitpolymul_2_128( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned n_64 );

void bitpolymul_2_64( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned n_64 );


#ifdef  __cplusplus
}
#endif


#endif
