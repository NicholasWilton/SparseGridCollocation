#pragma once

#include "stdafx.h"
#include "Params.h"
#include "Option.h"
#include "BasketOption.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Matrix2d;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using Eigen::Map;
using Eigen::aligned_allocator;
using namespace std;

namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API Algorithm
		{
		public:

			Algorithm();
			~Algorithm();

			double RootMeanSquare(VectorXd v);

			vector<MatrixXd> SIKc(int upper, int lower, Params p);
			vector<MatrixXd> SIKc(int upper, int lower, Params p, map<string, vector<vector<MatrixXd>>>& interpolation);

			vector<MatrixXd> MuSIKc(int upper, int lower, Params p);
			vector<MatrixXd> MuSIKc(int upper, int lower, Params p, map<string, vector<vector<MatrixXd>>>& interpolation);

			vector<MatrixXd> MuSIKcND(int upper, int lower, BasketOption option, Params p);
			vector<MatrixXd> MuSIKcND(int upper, int lower, BasketOption option, Params p, map<string, vector<vector<MatrixXd>>>& interpolation);

			map<string, vector<vector<MatrixXd>>> GetInterpolationState();


		private:

			//key - index from matlab code eg lamb3, TX3 etc -> '3'
			//value - vector of items lamb, TX, C, A
			//		- each in turn is a vector of Matrices
			map<string, vector<vector<MatrixXd>>> InterpolationState;
		};
	}
}