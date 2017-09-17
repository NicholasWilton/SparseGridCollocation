#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thrust/random.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include ".\..\include\eigen-eigen-67e894c6cd8f\Eigen\StdVector"
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#define API _declspec(dllexport)

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;
using namespace thrust;

namespace Leicester
{
	namespace ThrustLib
	{
		class API Gaussian
		{
		public:
			Gaussian(MatrixXd testNodes, MatrixXd centralNodes);
			Gaussian(MatrixXd testNodes);
			~Gaussian();
			static vector<MatrixXd> Gaussian2d(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			vector<MatrixXd> Gaussian2d(const MatrixXd &A, const MatrixXd &C);
			vector<MatrixXd> Gaussian2d(const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
		private:
			thrust::device_vector<double> testNodes;
			thrust::device_vector<double> centralNodes;
			int rows;
			int cols;
			
		};

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