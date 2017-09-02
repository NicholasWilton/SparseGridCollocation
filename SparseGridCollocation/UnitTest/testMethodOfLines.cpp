#include "stdafx.h"
#include "CppUnitTest.h"
#include "SparseGridCollocation.h"
#include "SmoothInitialX.h"
#include "SmoothInitialU.h"
#include <Eigen/Dense>
#include "Math.h"
#include "testCommon.h"
#include "Common.h"
#include "MoL.h"
#include "InterTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Eigen;
using namespace Leicester;

namespace UnitTest
{
	TEST_CLASS(testMuSIKc)
	{
	public:


		TEST_METHOD(testMoL)
		{
			
			
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

			vector<VectorXd> result = MoL::MethodOfLines(T, Tdone, Tend, dt, K, r, sigma, theta, inx1, inx2);

			VectorXd uX = SmoothInitialX::X();
			VectorXd uU = SmoothInitialU::U();

			Assert::IsTrue(testCommon::checkMatrix(uX, result[0], 0.0000001));
			Assert::IsTrue(testCommon::checkMatrix(uU, result[1], 0.0000001));
		

		}

		TEST_METHOD(testMoLND)
		{


			double T = 1.0;
			double Tdone = 0.0;
			double Tend = 0.2 * T;
			double dt = 1.0 / 10000.0;
			double K = 100.0;
			double r = 0.03;
			double sigma = 0.15;
			double theta = 0.5;
			VectorXd inx1(1);
			inx1[0] = -K;
			VectorXd inx2(1);
			inx2[0] = 6.0 * K;
			MatrixXd Corr = MatrixXd::Zero(1, 1);
			vector<vector<VectorXd>> result = MoL::MethodOfLinesND(T, Tdone, Tend, dt, K, r, sigma, theta, inx1, inx2, Corr);

			VectorXd uX = SmoothInitialX::X();
			VectorXd uU = SmoothInitialU::U();

			Assert::IsTrue(testCommon::checkMatrix(uX, result[0][0], 0.0000001));
			Assert::IsTrue(testCommon::checkMatrix(uU, result[0][1], 0.0000001));


		}

		TEST_METHOD(testMoLND_ODE)
		{


			double T = 1.0;
			double Tdone = 0.0;
			double Tend = 0.2 * T;
			double dt = 1.0 / 10000.0;
			double K = 100.0;
			double r = 0.03;
			double sigma = 0.15;
			double theta = 0.5;
			VectorXd inx1(1);
			inx1[0] = -K;
			VectorXd inx2(1);
			inx2[0] = 6.0 * K;
			MatrixXd Corr = MatrixXd::Zero(1, 1);
			vector<vector<VectorXd>> result = MoL::MethodOfLinesND_ODE(T, Tdone, Tend, dt, K, r, sigma, theta, inx1, inx2, Corr);

			VectorXd uX = SmoothInitialX::X();
			VectorXd uU = SmoothInitialU::U();

			Assert::IsTrue(testCommon::checkMatrix(uX, result[0][0], 0.0000001));
			Assert::IsTrue(testCommon::checkMatrix(uU, result[0][1], 0.0000001));


		}
	
	};
}