#include "stdafx.h"
#include "CppUnitTest.h"
#include "SparseGridCollocation.h"
#include "SmoothInitialX.h"
#include "SmoothInitialU.h"
#include <Eigen/Dense>
#include "Math.h"
#include "testCommon.h"
#include "Common.h"
#include "InterTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Eigen;


namespace UnitTest
{
	TEST_CLASS(testMuSIKc)
	{
	public:


		TEST_METHOD(testMoL)
		{
			SparseGridCollocation* test = new SparseGridCollocation();
			
			double T = 1.0;
			double Tdone = 0.0;
			double Tend = 0.2 * T;
			double dt = 1.0 / 10000.0;
			double K = 100.0;
			double r = 0.03;
			double sigma = 0.15;
			double theta = 0.5;
			double inx1 = -K;
			double inx2 = 6.0 * K;

			vector<VectorXd> result = test->MethodOfLines(T, Tdone, Tend, dt, K, r, sigma, theta, inx1, inx2);

			VectorXd uX = SmoothInitialX::X();
			VectorXd uU = SmoothInitialU::U();

			Assert::IsTrue(testCommon::checkMatrix(uX, result[0], 0.0000001));
			Assert::IsTrue(testCommon::checkMatrix(uU, result[1], 0.0000001));
		

		}

	
	};
}