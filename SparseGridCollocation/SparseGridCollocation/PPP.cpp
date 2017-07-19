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

double PPP::Calculate(RowVectorXd X)
{
	VectorXd x1 = SmoothInitialX::X();
	VectorXd U1 = SmoothInitialU::U();
	VectorXd u = (x1.array() == X(1)).select(U1, 0);
	double U = u.sum();

	if (U == 0)
		U = U1.trace();

	return U;
}