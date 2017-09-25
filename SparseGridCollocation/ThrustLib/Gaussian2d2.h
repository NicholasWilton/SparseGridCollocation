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
#include <map>

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
		class Gaussian2d2
		{
		public:
			Gaussian2d2();
			~Gaussian2d2();

			Gaussian2d2(MatrixXd testNodes, MatrixXd centralNodes);
			Gaussian2d2(MatrixXd testNodes);

			vector<MatrixXd> Gaussian2d_2(double tLower, double tUpper, double N[], const MatrixXd & A, const MatrixXd & C);

		private:
			thrust::device_vector<double> testNodes;
			thrust::device_vector<double> centralNodes;
			int rows;
			int cols;
		};
	}
}

