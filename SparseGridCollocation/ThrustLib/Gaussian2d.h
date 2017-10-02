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
#include "Functors.h"
#include "Common.h"
#include "Utility.h"



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
			Gaussian();
			Gaussian(MatrixXd testNodes, MatrixXd centralNodes);
			Gaussian(MatrixXd testNodes);
			Gaussian(int b, int d, double tLower, double tUpper, double xLower, double xHigher);
			~Gaussian();
			static vector<MatrixXd> Gaussian2d(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			vector<MatrixXd> Gaussian2d(const MatrixXd &A, const MatrixXd &C);
			vector<MatrixXd> Gaussian2d(const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			vector<MatrixXd> Gaussian2d_1(const MatrixXd & A, const MatrixXd & C, int count);
			vector<MatrixXd> Gaussian2d_2(double tLower, double tUpper, double N[], const MatrixXd & A, const MatrixXd & C);

		private:
			thrust::device_vector<double> testNodes;
			thrust::device_vector<double> centralNodes;
			int rows;
			int cols;

			struct nodesDetails
			{
				int rows;
				int cols;
				device_vector<double> nodes;
			};

			map<int, nodesDetails> nodeMap;
			void subnumber(int b, int d, double matrix[]);
			void GetN(int b, int d, double N[]);
			
		};

	


	}
}