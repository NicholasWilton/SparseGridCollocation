#include "stdafx.h"
#include "CppUnitTest.h"
#include "SparseGridCollocation.h"
#include "windows.h"
#include "testCommon.h"
#include "Common.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>



using Eigen::Matrix2d;

using namespace Microsoft::VisualStudio::CppUnitTestFramework;



namespace UnitTest
{
	TEST_CLASS(testInterpolation)
	{
	public:


		TEST_METHOD(TestInterpolate)
		{
			SparseGridCollocation* test = new SparseGridCollocation();
			
			vector<vector<MatrixXd>> result = test->interpolate(2, 0.865, 3, 2, 0, 300, 0.03, 0.15, 1, 100);

			
			//Assert::IsTrue(testCommon::checkMatrix(expected, result));
		}

		TEST_METHOD(TestSubnumber)
		{
			SparseGridCollocation* test = new SparseGridCollocation();

			MatrixXd expected(2, 2);
			expected << 1, 2, 2, 1;
			MatrixXd result = test->subnumber(3,2);

			Assert::IsTrue(testCommon::checkMatrix(expected, result));

		}



	};
}