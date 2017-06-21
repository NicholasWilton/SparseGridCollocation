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
	TEST_CLASS(testShapeLambda)
	{
	public:
		

		TEST_METHOD(TestPrintMatrix)
		{


			SparseGridCollocation* test = new SparseGridCollocation();
			MatrixXd uLamb(7, 2);
			uLamb << 168.177040362907, -384.857133569469, 437.285285931294, -355.212798882940, 443.385219414521, -109.178371522743, 221.964431301908, -185.896683167106, 91.7816691444462, -271.563020925667, 179.474487920222, -412.630669834898, 475.510735359416, -393.047823887079;
			wstring s = Common::printMatrix(uLamb);
						
			Logger::WriteMessage(s.c_str());
		}

		TEST_METHOD(TestShapeLambda2D)
		{
			SparseGridCollocation* test = new SparseGridCollocation();
			int ch = 10000;
			double inx1 = 0;
			double inx2 = 300;
			VectorXd x = VectorXd::LinSpaced(ch, inx1, inx2);
			VectorXd t(ch);
			t.fill(0);

			
			Matrix<double, 2, 2> N;
			N << 3, 5, 5, 3;

			MatrixXd uLamb(15, 1);
			uLamb << 168.177040362907, -384.857133569469, 437.285285931294, -355.212798882940, 443.385219414521, -109.178371522743, 221.964431301908, -185.896683167106, 91.7816691444462, -271.563020925667, 179.474487920222, -412.630669834898, 475.510735359416, -393.047823887079, 457.473155903381;
			MatrixXd uTX = MatrixXd(15, 2);
			uTX << 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.4325, 0, 0.4325, 75, 0.4325, 150, 0.4325, 225, 0.4325, 300, 0.865, 0, 0.865, 75, 0.865, 150, 0.865, 225, 0.865, 300;
			MatrixXd uC(1, 2);
			uC << 1.73000000000000, 600;
			MatrixXd uA(1, 2);
			uA << 2, 4;

			
			//TODO: improve test to iterate more than once:
			//for (int i = 0; i < N.cols(); i++)
			for (int i = 0; i < 1; i++)
			{
				double coef = 2;
				double tsec = 0.8650L;
				double r = 0.03;
				double sigma = 0.15;
				double T = 1;
				double E = 100;
				double inx1 = 0;
				double inx2 = 300;
				
				vector<MatrixXd> result = test->shapelambda2D(coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i));
				
				
				MatrixXd l = result[0];
				Assert::IsTrue(testCommon::checkMatrix(uLamb,  l, 1));
				MatrixXd tx = result[1];
				Assert::IsTrue(testCommon::checkMatrix(uTX, tx));
				MatrixXd c = result[2];
				Assert::IsTrue(testCommon::checkMatrix(uC, c));
				MatrixXd a = result[3];
				Assert::IsTrue(testCommon::checkMatrix(uA, a));
			}
		}



	};
}