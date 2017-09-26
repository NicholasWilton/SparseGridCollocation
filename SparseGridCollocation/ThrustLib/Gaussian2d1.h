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
		class API Gaussian2d1
		{
		public:
			Gaussian2d1(MatrixXd testNodes, MatrixXd centralNodes);
			Gaussian2d1(MatrixXd testNodes);
			~Gaussian2d1();
			static vector<MatrixXd> Gaussian2d(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			vector<MatrixXd> Gaussian2d(const MatrixXd &A, const MatrixXd &C);
			vector<MatrixXd> Gaussian2d(const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
		private:
			thrust::device_vector<double> testNodes;
			thrust::device_vector<double> centralNodes;
			int rows;
			int cols;
		};
	}
}