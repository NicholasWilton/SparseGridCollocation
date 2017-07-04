#include "stdafx.h"
#include "Common.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>



using namespace Eigen;
using namespace std;

#ifdef UNITTEST
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
#endif // UNITTEST

Common::Common()
{
}


Common::~Common()
{
}

void Common::Logger(string message)
{
#ifdef UNITTEST
	Logger::WriteMessage(message);
#endif // UNITTEST
	cout << message << endl;
}

wstring Common::printMatrix(MatrixXd m)
{
	int cols = m.cols();
	int rows = m.rows();

	const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

	wstringstream ss;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			ss << m(i, j) << "\t";

		}
		ss << "\r\n";
	}

	return ss.str();
}

double Common::prod(vector<double> x)
{
	double prod = 1.0;

	for (unsigned int i = 0; i < x.size(); i++)
		prod *= x[i];
	return prod;
}


vector<double> Common::linspace(double a, double b, size_t N)
{
	double h = (b - a) / (N - 1);
	vector<double> xs(N);
	typename vector<double>::iterator x;
	double val;
	for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
		*x = val;
	return xs;
}

bool Common::checkMatrix(MatrixXd expected, MatrixXd actual)
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

				cout << message << endl;

				result = false;
			}
		}

	return result;
}

bool Common::checkMatrix(MatrixXd expected, MatrixXd actual, double precision)
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

				cout << message << endl;

				result = false;
			}
		}

	return result;
}