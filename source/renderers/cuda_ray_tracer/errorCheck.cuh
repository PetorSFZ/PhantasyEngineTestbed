/************************************************************/
/*!	\brief CUDA Helpers
 */
/* Copyright (c) 2010, 2011: Markus Billeter
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/********************************************************************/
/*
* Modified by Viktor Kämpe to use std::cout instead of fprintf
* Modified by Peter Hillerström to use fprintf again
*/

#ifndef HELPERS_CUH_EF82B87F_07AB_498D_AAD1_BC8432BE97B9
#define HELPERS_CUH_EF82B87F_07AB_498D_AAD1_BC8432BE97B9

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECKED_CALL ::detail::CudaErrorChecker(__LINE__,__FILE__) =
#define CUDA_CHECK_ERROR() \
	{  \
		cudaThreadSynchronize(); \
		cudaError_t err = cudaGetLastError(); \
		if( cudaSuccess != err ) { \
			std::fprintf(stderr, "%i:%i: cuda state error %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		} \
	} \
	/*EOM*/

namespace detail
{
	struct CudaErrorChecker
	{
		int line;
		const char* file;

		inline CudaErrorChecker( int aLine, const char* aFile ) : line(aLine), file(aFile){}

		inline cudaError_t operator=( cudaError_t err ) {
			if( cudaSuccess != err ) {
				std::fprintf(stderr, "%s:%i: cuda state error %s\n", file, line, cudaGetErrorString(err));
			}

			return err;
		}
	};
}

#endif // HELPERS_CUH_EF82B87F_07AB_498D_AAD1_BC8432BE97B9
