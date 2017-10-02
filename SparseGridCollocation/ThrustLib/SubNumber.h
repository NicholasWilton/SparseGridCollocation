#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <thrust/random.h>
//#include <thrust/system/cuda/execution_policy.h>
//#include <thrust/iterator/counting_iterator.h>
//#include <thrust/functional.h>
//#include <thrust/transform_reduce.h>
//#include <thrust/device_vector.h>
//#include <iostream>
//#include <iomanip>
//#include <cmath>
//#include ".\..\include\eigen-eigen-67e894c6cd8f\Eigen\StdVector"
//#include <thrust/system/cuda/experimental/pinned_allocator.h>
//#include <map>
//#include "cuda_runtime.h"
#define API _declspec(dllexport)

//using Eigen::MatrixXd;
//using Eigen::VectorXd;
//using namespace std;
//using namespace Eigen;
//using namespace thrust;
namespace Leicester
{
	namespace ThrustLib
	{
		//class API SubNumber
		//{
		//public:
			//__device__ extern void subnumber(int b, int d, double *matrix);
			//__global__ extern void Add_CUDA(int b, int d, double *N);
		//};

		struct matrixDim
		{
			int rows;
			int cols;
		};
	}
}
