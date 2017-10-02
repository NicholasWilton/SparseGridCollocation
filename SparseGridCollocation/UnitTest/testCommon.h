#pragma once

using Eigen::MatrixXd;

namespace UnitTest
{
	class API testCommon
	{
	public:
		testCommon();
		~testCommon();
		static bool checkMatrix(MatrixXd reference, MatrixXd actual);
		static bool checkMatrix(MatrixXd reference, MatrixXd actual, double precision);
		static MatrixXd LoadTX();
	};

}