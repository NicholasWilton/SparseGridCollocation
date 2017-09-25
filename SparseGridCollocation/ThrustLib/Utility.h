#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include ".\..\include\eigen-eigen-67e894c6cd8f\Eigen\StdVector"
#include <thrust/system/cuda/experimental/pinned_allocator.h>


#define API _declspec(dllexport)

using namespace std;
using namespace Eigen;

namespace Leicester
{
	namespace ThrustLib
	{
		class API Utility
		{
		public:
			static void printMatrix(const double *matrix, dim3 dimMatrix)
			{
				int mSize = sizeof(matrix);

				printf("printing matrix data=");
				for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
					printf("%f,", matrix[x]);
				printf("\r\n");
				printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);

				for (int y = 0; y < dimMatrix.y; y++)
				{
					for (int x = 0; x < dimMatrix.x; x++)
					{
						int idx = (x * dimMatrix.y) + y;
						printf("%.16f ", matrix[idx]);
					}
					printf("\r\n");
				}
			};

			static wstring printMatrix(MatrixXd m)
			{
				int cols = m.cols();
				int rows = m.rows();

				wstringstream ss;
				ss << setprecision(25);
				for (int i = 0; i < rows; i++)
				{
					for (int j = 0; j < cols; j++)
					{
						double d = m(i, j);
						ss << d << "\t";

					}
					ss << "\r\n";
				}

				return ss.str();
			};
		};
	}
}