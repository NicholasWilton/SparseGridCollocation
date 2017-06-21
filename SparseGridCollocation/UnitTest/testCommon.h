#pragma once

using Eigen::MatrixXd;

namespace UnitTest
{
	class testCommon
	{
	public:
		testCommon();
		~testCommon();
		static bool testCommon::checkMatrix(MatrixXd reference, MatrixXd actual);
		static bool testCommon::checkMatrix(MatrixXd reference, MatrixXd actual, double precision);
	};

}