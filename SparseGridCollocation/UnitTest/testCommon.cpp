#include "stdafx.h"
#include "testCommon.h"

#include "SparseGridCollocation.h"
#include <Eigen/Dense>
#include "Math.h"
#include "CppUnitTest.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Eigen;
using namespace UnitTest;

testCommon::testCommon()
{
}


testCommon::~testCommon()
{
}

MatrixXd testCommon::LoadTX()
{
	ifstream infile("TX.txt");
	vector<double> doubles;
	int count = 0;
	while (infile)
	{
		string s;
		if (!getline(infile, s)) break;
		
		stringstream ss(s);
		while (ss.good())
		{
			string substr;
			getline(ss, substr, ',');
			doubles.push_back(stod(substr));
		}
		
		//doubles.push_back(stod(s));
		count++;
	}

	MatrixXd U(doubles.size()/2, 2);
	vector<double>::iterator it;
	int i = 0;
	for (it = doubles.begin(); it < doubles.end(); it++, i++) {
		if (i % 2 == 0)
			U(i/2,0) = doubles[i];
		else
			U(floor(i/2),1) = doubles[i];
	}

	return U;
}

bool testCommon::checkMatrix(MatrixXd expected, MatrixXd actual)
{
	bool result = true;
	int cols = expected.cols();
	int rows = expected.rows();
	wchar_t message[20000];

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			double diff = abs((expected(i, j) - actual(i, j)));

			if (diff > DBL_EPSILON)
			{
				const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

				_swprintf(message, L"%g != %g index[%i,%i]", expected(i, j), actual(i, j), i, j);

				Logger::WriteMessage(message);

				result = false;
			}
		}

	return result;
}

bool testCommon::checkMatrix(MatrixXd expected, MatrixXd actual, double precision)
{
	bool result = true;
	int cols = expected.cols();
	int rows = expected.rows();
	wchar_t message[20000];

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			double diff = abs((expected(i, j) - actual(i, j)));
			if (diff > precision)
			{
				const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

				_swprintf(message, L"%g != %g index[%i,%i]", expected(i, j), actual(i, j), i, j);

				Logger::WriteMessage(message);

				result = false;
			}
		}

	return result;
}
