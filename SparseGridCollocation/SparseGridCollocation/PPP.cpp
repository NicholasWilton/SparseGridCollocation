#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialU.h"
#include "SmoothInitialX.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using namespace std;

double PPP::Calculate()
{
	//A = load('Smoothinitial');
	
	//x1 = A.X_ini;
	VectorXd x1 = SmoothInitialX::X();
	//U1 = A.U_ini;
	VectorXd U1 = SmoothInitialU::U();

	//I = X(1, 2) == x1;
	//not sure about this:
	 //VectorXd I = X(1, 2) == x1;

	//U = U1(I);
	 //could be wrong:
	 double U = U1.sum();

	 return U;
}