// TestHarness.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
//#include "CppUnitTest.h"
//#include "SparseGridCollocation.h"
//#include <Eigen/Dense>
#include "Math.h"

#include "Common.h"
using namespace Eigen;
using namespace std;


int main()
{
	SparseGridCollocation* test = new SparseGridCollocation();

	VectorXd uRMS(9);
	uRMS << 3.74273075820591, 1.39067846770261, 0.436408235898941, 0.122815017267700, 0.0328347672816514, 0.00835749999263466, 0.00221774306858896, 0.000597422702846034, 0.000162823604659233;
	VectorXd uMax = VectorXd(9);
	uMax << 6.55956979634471, 3.70216720531096, 1.42734558657324, 0.416600021955134, 0.114733914477483, 0.0293262625138198, 0.00765938580167358, 0.00198337219728728, 0.000513094949184278;

	vector<VectorXd> result = test->MuSIKGeneric();

	cout << Common::printMatrix(result[0]).c_str() << endl;
	
	if (!Common::checkMatrix(uRMS, result[0], 0.0000001))
		cout << "check RMS failed" << endl;


	cout << Common::printMatrix(result[0]).c_str() << endl;

	if(!Common::checkMatrix(uMax, result[1], 0.0000001))
		cout << "check MAX failed" << endl;

    return 0;
}

