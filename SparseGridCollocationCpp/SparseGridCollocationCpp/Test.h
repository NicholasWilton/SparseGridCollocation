#pragma once
#include "stdafx.h"

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
		class API Test
		{
		public:
			Test();
			~Test();

			static double inner(double t, double x, const vector<MatrixXd> &lamb, const vector<MatrixXd> &TX, const vector<MatrixXd> &C, const vector<MatrixXd> &A);
			static double innerND(MatrixXd TXYZ, const vector<MatrixXd> &lamb, const vector<MatrixXd> &TX, const vector<MatrixXd> &C, const vector<MatrixXd> &A);

			static VectorXd inter(MatrixXd X, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);


		};
	}
}


