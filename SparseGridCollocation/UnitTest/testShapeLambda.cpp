#include "stdafx.h"
#include "CppUnitTest.h"
#include "SparseGridCollocation.h"
#include "windows.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest
{		
	TEST_CLASS(testShapeLambda)
	{
	public:
		
		bool checkMatrix(MatrixXd reference, MatrixXd actual)
		{
			
			bool result = true;
			int cols = reference.cols();
			int rows = reference.rows();
			//int size = 200 * cols * rows;
			wchar_t message[20000];
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
				{
					if (reference(i, j) != actual(i, j))
					{
						
						_swprintf(message, L"%g != %g index[%g,%g] /r/n", reference(i, j), actual(i, j), i, j);
						//cout << reference(i, j) << "!=" << actual(i, j) << " index[" << i << "," << j << "]" << endl;
						result = false;
					}
				}
			if (!result)
				Assert::Fail(message, LINE_INFO());

			return result;
		}

		bool checkMatrix(MatrixXd* reference, MatrixXd* actual)
		{
			bool result = true;
			int cols = reference->cols();
			int rows = reference->rows();
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
				{
					MatrixXd r = *reference;
					MatrixXd a = *actual;
					if (r(i, j) != a(i, j))
						result = false;
				}
			return result;
		}

		TEST_METHOD(TestPrintMatrix)
		{


			SparseGridCollocation* test = new SparseGridCollocation();
			MatrixXd uLamb(7, 2);
			uLamb << 168.177040362907, -384.857133569469, 437.285285931294, -355.212798882940, 443.385219414521, -109.178371522743, 221.964431301908, -185.896683167106, 91.7816691444462, -271.563020925667, 179.474487920222, -412.630669834898, 475.510735359416, -393.047823887079;
			wstring s = test->printMatrix(&uLamb);
						
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

			MatrixXd lamb(15, 1);
			MatrixXd TX(ch, 2);
			TX << t, x;
			//MatrixXd *TX3 = new MatrixXd(0,0);
			MatrixXd C(2, 1);
			MatrixXd A(2, 1);
			Matrix<double, 2, 2> N;
			N << 3, 5, 5, 3;

			MatrixXd uLamb(15, 1);
			uLamb << 168.177040362907, -384.857133569469, 437.285285931294, -355.212798882940, 443.385219414521, -109.178371522743, 221.964431301908, -185.896683167106, 91.7816691444462, -271.563020925667, 179.474487920222, -412.630669834898, 475.510735359416, -393.047823887079, 457.473155903381;
			MatrixXd uTX = MatrixXd(15, 2);
			uTX << 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.4325, 0, 0.4325, 75, 0.4325, 150, 0.4325, 225, 0.4325, 300, 0.865, 0, 0.865, 75, 0.865, 1510, 0.865, 225, 0.865, 300;
			MatrixXd uC(2, 1);
			uC << 1.73000000000000, 600;
			MatrixXd uA(2, 1);
			uA << 2, 4;

			for (int i = 0; i < 2; i++)
			{
				double coef = 2;
				double tsec = 0.8650;
				double r = 0.03;
				double sigma = 0.15;
				double T = 1;
				double E = 100;
				double inx1 = 0;
				double inx2 = 300;
				test->shapelambda2D(lamb, TX, C, A, coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i));
				bool test = true;
				Assert::IsTrue(checkMatrix(uLamb, lamb));
				Assert::IsTrue(checkMatrix(uTX, TX));
				Assert::IsTrue(checkMatrix(uC, C));
				Assert::IsTrue(checkMatrix(uA, A));
			}
		}



	};
}