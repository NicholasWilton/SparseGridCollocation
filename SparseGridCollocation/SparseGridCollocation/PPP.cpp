#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialU.h"
#include "SmoothInitialX.h"
#include "SparseGridCollocation.h"
#include "Common.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using namespace std;

double PPP::Calculate(const RowVectorXd &X)
{
	VectorXd x1 = SmoothInitialX::X();
	VectorXd U1 = SmoothInitialU::U();

	//TODO: X(1) should this be X(0) or X(last)?
	VectorXd u = (x1.array() == X(1)).select(U1, 0);
	double U = u.sum();

	if (U == 0)
		U = U1.trace();

	return U;
}