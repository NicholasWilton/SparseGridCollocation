#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialU.h"
#include "SmoothInitialX.h"
#include "SparseGridCollocation.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using namespace std;

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

double PPP::Calculate(RowVectorXd X)
{
	Logger::WriteMessage(SparseGridCollocation::printMatrixI(X).c_str());
	//A = load('Smoothinitial');

	//x1 = A.X_ini;
	VectorXd x1 = SmoothInitialX::X();
	Logger::WriteMessage(SparseGridCollocation::printMatrixI(x1).c_str());
	//U1 = A.U_ini;
	VectorXd U1 = SmoothInitialU::U();
	Logger::WriteMessage(SparseGridCollocation::printMatrixI(U1).c_str());


	//I = X(1, 2) == x1;
	//U = U1(I);

	//If both SmoothInitial files have different numbers of lines then this code will fail
	VectorXd u = (x1.array() == X(1)).select(U1, 0);
	Logger::WriteMessage(SparseGridCollocation::printMatrixI(u).c_str());
	double U = u.sum();

	//this is a bit of a hack to mimic the original behaviour:
	if (U == 0)
		U = U1.trace();

	return U;
}