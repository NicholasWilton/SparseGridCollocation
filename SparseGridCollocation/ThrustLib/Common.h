#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/random.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#define API _declspec(dllexport)


using namespace std;

namespace Leicester
{

	namespace ThrustLib
	{
		typedef thrust::host_vector<double, thrust::cuda::experimental::pinned_allocator<double>> pinnedVector;

		struct API MemoryInfo
		{
			int total;
			int free;
		};

		class API Common
		{
		public:
			static MemoryInfo GetMemory()
			{
				size_t free_bytes;

				size_t total_bytes;

				cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes);

				MemoryInfo res;
				res.free = free_bytes;
				res.total = total_bytes;

				return res;
			}
		};



		//void
		//	printMatrixFormatted(const double *matrix, dim3 dimMatrix)
		//{
		//	int mSize = sizeof(matrix);

		//	printf("printing matrix data=");
		//	for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
		//		printf("%f,", matrix[x]);
		//	printf("\r\n");
		//	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);

		//	for (int y = 0; y < dimMatrix.y; y++)
		//	{
		//		for (int x = 0; x < dimMatrix.x; x++)
		//		{
		//			int idx = (x * dimMatrix.y) + y;
		//			//if ( mSize > idx)
		//			printf("indx=%i value=%f\t", idx, matrix[idx]);
		//		}
		//		printf("\r\n");
		//	}
		//}

		//void
		//	printMatrix(const double *matrix, dim3 dimMatrix)
		//{
		//	int mSize = sizeof(matrix);

		//	printf("printing matrix data=");
		//	for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
		//		printf("%f,", matrix[x]);
		//	printf("\r\n");
		//	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);

		//	for (int y = 0; y < dimMatrix.y; y++)
		//	{
		//		for (int x = 0; x < dimMatrix.x; x++)
		//		{
		//			int idx = (x * dimMatrix.y) + y;
		//			printf("%f ", matrix[idx]);
		//		}
		//		printf("\r\n");
		//	}
		//}
	}
}
