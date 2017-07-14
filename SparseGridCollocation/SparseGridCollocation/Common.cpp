#include "stdafx.h"
#include "Common.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>


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

void Common::Logger(wstring message)
{
#if defined(UNITTEST)
	Logger::WriteMessage(message);
#endif // UNITTEST
	wcout << message << endl;
}

wstring Common::printMatrix(MatrixXd m)
{
	int cols = m.cols();
	int rows = m.rows();
	
	wstringstream ss;
	ss << setprecision(25);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double d = m(i, j);
			ss << d << "\t";

		}
		ss << "\r\n";
	}

	return ss.str();
}

wstring Common::printMatrixHexW(MatrixXd m)
{
	int cols = m.cols();
	int rows = m.rows();

	wstringstream ss;
	ss << setprecision(25);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double d = m(i, j);
			ss << double2hexstrW(d) << "\t";

		}
		ss << "\r\n";
	}

	return ss.str();
}

string Common::printMatrixHexA(MatrixXd m)
{
	int cols = m.cols();
	int rows = m.rows();

	stringstream ss;
	ss << setprecision(25);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double d = m(i, j);
			ss << double2hexstrA(d) << "\t";

		}
		ss << "\r\n";
	}

	return ss.str();
}

std::wstring Common::double2hexstrW(double x) {

	union
	{
		long long i;
		double    d;
	} value;

	value.d = x;

	std::wostringstream buf;
	buf << std::hex << std::setfill(L'0') << std::setw(16) << value.i;

	return buf.str();

}

std::string Common::double2hexstrA(double d) {

	char buffer[25] = { 0 };

	::snprintf(buffer, 25, "%A", d); // TODO Check for errors

	return buffer;
}

double Common::hexastr2doubleA(const std::string& s) {

	double d = 0.0;

	::sscanf(s.c_str(), "%lA", &d); // TODO Check for errors

	return d;
}

string Common::printMatrixA(MatrixXd m)
{
	int cols = m.cols();
	int rows = m.rows();

	IOFormat CleanFmt(FullPrecision, 0, ", ", "\n", "[", "]");

	stringstream ss;

	ss << m.format(CleanFmt);

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

				wcout << message << endl;

				result = false;
			}
		}

	return result;
}