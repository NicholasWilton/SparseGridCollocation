#include "stdafx.h"
#include "testCommon.h"

#include "SparseGridCollocation.h"
#include <Eigen/Dense>
#include "Math.h"
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Eigen;
using namespace UnitTest;

testCommon::testCommon()
{
}


testCommon::~testCommon()
{
}

bool testCommon::checkMatrix(MatrixXd reference, MatrixXd actual)
{
	bool result = true;
	int cols = reference.cols();
	int rows = reference.rows();
	wchar_t message[20000];

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			double diff = abs((reference(i, j) - actual(i, j)));

			if (diff > DBL_EPSILON)
			{
				const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

				_swprintf(message, L"%g != %g index[%i,%i]", reference(i, j), actual(i, j), i, j);

				Logger::WriteMessage(message);

				result = false;
			}
		}

	return result;
}

bool testCommon::checkMatrix(MatrixXd reference, MatrixXd actual, double precision)
{
	bool result = true;
	int cols = reference.cols();
	int rows = reference.rows();
	wchar_t message[20000];

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			double diff = abs((reference(i, j) - actual(i, j)));
			if (diff > precision)
			{
				const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

				_swprintf(message, L"%g != %g index[%i,%i]", reference(i, j), actual(i, j), i, j);

				Logger::WriteMessage(message);

				result = false;
			}
		}

	return result;
}
