#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_functions.h>
#include "SubNumber.h"

#define API _declspec(dllexport)



namespace Leicester
{
	namespace ThrustLib
	{


		//class API NodeRegistry
		//{
		//public:
		//	NodeRegistry() {};
		//	~NodeRegistry() {};
		//	map<string, device_vector<double>> Ns;
		//	void Add(int b, int d) {
		//		
		//		double* d_N, h_N;
		//		cudaError_t e = cudaMalloc((void **)&d_N, sizeof(double));
		//		
		//		//Add_CUDA << <1, 1 >> > (b, d, d_N);

		//		//e = cudaMemcpy(h_N, d_N, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		//		device_ptr<double> dp_N = device_pointer_cast<double>(d_N);
		//		device_vector<double> N(dp_N, dp_N + d_N[0] * d_N[1]);
		//		stringstream ss;
		//		ss << b << "," << d;
		//		Ns.insert_or_assign(ss.str(), N);

		//	};
		//	void Remove() {};
		//	void Get() {};
		//	
		////private:
		//	


		//	map<std::string, thrust::device_vector<double>> nodes;
		//	MatrixXd primeNMatrix(int b, int d);
		//	MatrixXd subnumber(int b, int d);
		//	//static matrixDim subnumber(int b, int d, double *matrix);

		//};

		extern __global__ void GenerateTestNodes(double timeLowerLimit, double timeUpperLimit, double N[], double TXYZ[]);
		extern __global__ void BuildRegistry(int b, int d, double tLower, double tUpper, double xLower, double xHigher, double** nodes);

	}
}
