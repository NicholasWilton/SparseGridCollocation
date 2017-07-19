#include "stdafx.h"
#include "Common.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <windows.h>
#include <direct.h>
#include <map>
//#include "../include/boost_1_64_0/boost/multiprecision/cpp_dec_float.hpp"

using namespace Eigen;
using namespace std;
//using namespace boost::multiprecision;

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
				//const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

				//_swprintf(message, L"%f != %f index[%i,%i]", expected(i, j), actual(i, j), i, j);
				//wcout << message << endl;

				wcout << setprecision(25) << expected(i, j) << " != " << actual(i, j) << " index[" << i, ',' << j << ']';
				wcout << endl;

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
				//const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

				//_swprintf(message, L"%f != %f index[%i,%i]", expected(i, j), actual(i, j), i, j);

				wcout << setprecision(25) << expected(i, j) << " != " << actual(i, j) << " index[" << i, ',' << j << ']';
				wcout << endl;

				result = false;
			}
		}

	return result;
}

bool Common::checkMatrix(MatrixXd expected, MatrixXd actual, double precision, bool print)
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
				//const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

				//_swprintf(message, L"%f != %f index[%i,%i]", expected(i, j), actual(i, j), i, j);

				if (print)
				{
					wcout << setprecision(25) << expected(i, j) << " != " << actual(i, j) << " index[" << i, ',' << j << ']';
					wcout << endl;
				}
				result = false;
			}
		}

	return result;
}

void Common::WriteToBinary(string fileName, MatrixXd matrix)
{
	char cCurrentPath[FILENAME_MAX];
	string path = _getcwd(cCurrentPath, sizeof(cCurrentPath));
	stringstream ss;
	ofstream fout;
	ss << path << "\\" << fileName;
	fout.open(ss.str(), ios::binary);
	cout << "Writing: " << ss.str() << endl;
	for (int i = 0; i < matrix.rows(); i++)
	{
		for (int j = 0; j < matrix.cols(); j++)
		{
			double d = matrix(i, j);
			fout.write((char *)(&d), sizeof(d));
		}
	}
	fout.close();
};

MatrixXd Common::ReadBinary(string fileName, int rows, int cols)
{

	double value;
	vector<double> read;
	char cCurrentPath[FILENAME_MAX];
	string path = _getcwd(cCurrentPath, sizeof(cCurrentPath));
	stringstream ss;
	ss << path << "\\" << fileName;
	cout << "Reading: " << ss.str() << endl;
	fstream kFile(ss.str(), ios::in | ios::out | ios::binary);
	kFile.seekg(0);

	while (EOF != kFile.peek())
	{
		kFile.read((char*)&value, sizeof(double));
		read.push_back(value);
	}

	MatrixXd result = MatrixXd::Zero(rows, cols);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result(i, j) = read[(i * result.cols()) + j];
		}
	}
	return result;
};

//typedef number<cpp_dec_float<14> > cpp_dec_float_14;
//typedef number<cpp_dec_float<15> > cpp_dec_float_15;
//typedef number<cpp_dec_float<16> > cpp_dec_float_16;
//typedef number<cpp_dec_float<17> > cpp_dec_float_17;
//typedef number<cpp_dec_float<18> > cpp_dec_float_18;
//typedef number<cpp_dec_float<19> > cpp_dec_float_19;
//typedef number<cpp_dec_float<20> > cpp_dec_float_20;
//
//MatrixXd Common::mult(MatrixXd &a, MatrixXd &b)
//{
//	MatrixXd result(a.rows(), b.cols());
//
//	//assume a & b are compatible
//	for (int i = 0; i < a.rows(); i++)
//		for (int j = 0; j < b.cols(); j++)
//		{
//			cpp_dec_float_16 sum = 0;
//			for (int x = 0; x < a.cols(); x++)
//			{
//				cpp_dec_float_16 l = a(i, x);
//				cpp_dec_float_16 r = b(x, j);
//				sum = sum + (l * r);
//
//
//			}
//			result(i, j) = (double)sum;
//		}
//	return result;
//
//};

std::map<string, double> Common::LoadMock()
{
	int n = 500000;
	//Data xx;
	int *key = new int[n];
	int *thread = new int[n];
	int *num = new int[n];
	int *sub = new int[n];
	double *value = new double[n];

	fstream kFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_key.dat", ios::in | ios::out | ios::binary);
	kFile.seekg(0);
	kFile.read((char*)key, sizeof(int) * n);

	fstream tFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_thread.dat", ios::in | ios::out | ios::binary);
	tFile.seekg(0);
	tFile.read((char*)thread, sizeof(int) * n);

	fstream nFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_num.dat", ios::in | ios::out | ios::binary);
	nFile.seekg(0);
	nFile.read((char*)num, sizeof(int) * n);

	fstream sFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_sub.dat", ios::in | ios::out | ios::binary);
	sFile.seekg(0);
	sFile.read((char*)sub, sizeof(int) * n);

	fstream vFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_value.dat", ios::in | ios::out | ios::binary);
	vFile.seekg(0);
	vFile.read((char*)value, sizeof(double) * n);

	map<string, double> result;

	for (int i = 0; i < n; i++)
	{
		stringstream ss;
		ss << key[i] << '_' << thread[i] << '_' << num[i] << '_' << sub[i];
		string s = ss.str();
		result[s] = value[i];
	}
	return result;
}

map<string, double> Common::mock;

double Common::innerMock(string key, int thread, int num, int sub)
{
	if (Common::mock.size() == 0)
		Common::mock = LoadMock();
	stringstream ss;
	int k;
	if (key == "4")
		k = 1;
	if (key == "_3")
		k = 2;
	if (key == "5")
		k = 3;
	if (key == "_4")
		k = 4;
	if (key == "6")
		k = 5;
	if (key == "_5")
		k = 6;
	if (key == "7")
		k = 7;
	if (key == "_6")
		k = 8;
	if (key == "8")
		k = 9;
	if (key == "_7")
		k = 10;
	if (key == "9")
		k = 11;
	if (key == "_8")
		k = 12;
	if (key == "10")
		k = 13;
	if (key == "_9")
		k = 14;
	if (key == "11")
		k = 15;
	if (key == "_10")
		k = 16;
	/*if (k >= 13)
	ss << k << '_' << thread+1 << '_' << num+1 << '_' << sub + 1;
	else*/
	ss << k << '_' << thread + 1 << '_' << num + 1 << '_' << sub;

	double r = mock[ss.str()];
	return r;
}

