#pragma once
//#include "cuda_runtime.h"
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
#include <map>
#include ".\..\include\eigen-eigen-67e894c6cd8f\Eigen\StdVector"
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "NodeRegistry.h"
#include "Functors.h"
#include "Common.h"
#include "Utility.h"

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
		class API Gaussian2d2
		{
		public:
			Gaussian2d2();
			~Gaussian2d2();

			Gaussian2d2(double tLower, double tUpper, const double* N);
			
			vector<MatrixXd> Gaussian2d(const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C);
			vector<MatrixXd> Gaussian2d(const MatrixXd & A, const MatrixXd & C);

		private:
			thrust::device_ptr<double> testNodes;
			thrust::device_ptr<double> centralNodes;
			int dimensions;
			int rows;
			int cols;
		};
	}
}

