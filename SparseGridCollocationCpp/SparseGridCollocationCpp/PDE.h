#pragma once
#include "stdafx.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API PDE
		{
		public:
			static int callCount;
			PDE();
			~PDE();
			static MatrixXd BlackScholes(const MatrixXd &node, double r, double sigma,
				const vector<MatrixXd> &lambda2, const vector<MatrixXd> &TX2, const vector<MatrixXd> &C2, const vector<MatrixXd> &A2,
				const vector<MatrixXd> &lambda3, const vector<MatrixXd> &TX3, const vector<MatrixXd> &C3, const vector<MatrixXd> A3);

			static MatrixXd BlackScholesC(const MatrixXd &node, double r, double sigma,
				const vector<MatrixXd> &lambda2, const vector<MatrixXd> &TX2, const vector<MatrixXd> &C2, const vector<MatrixXd> &A2,
				const vector<MatrixXd> &lambda3, const vector<MatrixXd> &TX3, const vector<MatrixXd> &C3, const vector<MatrixXd> A3);

			static MatrixXd BlackScholesNd(const MatrixXd &node, double r, double sigma, vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state);

			static MatrixXd BlackScholesNdC(const MatrixXd &node, double r, double sigma, vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state);

		};
	}
}
