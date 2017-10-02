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
		class API RBF
		{
		public:
			RBF();
			~RBF();
			static VectorXd exp(const VectorXd &v);
			static vector<MatrixXd> GaussianND_ODE(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			static vector<MatrixXd> GaussianND(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			static vector<MatrixXd> Gaussian2D(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			static vector<MatrixXd> MultiQuadric1D(const MatrixXd &xx, double x, double c);
			static vector<MatrixXd> MultiQuadric2D(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			static vector<vector<MatrixXd>> MultiQuadricND(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			static vector<MatrixXd> MultiQuadricND_ODE(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
			static VectorXd PhiProduct(vector<VectorXd> PhiN, int n);
		};
	}
}
