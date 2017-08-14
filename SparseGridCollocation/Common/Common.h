#pragma once


#include "stdafx.h"
#include <map>

using Eigen::MatrixXd;
using namespace Eigen;
using namespace std;


class API Common
{

public:

	Common();
	~Common();
	static wstring printMatrix(MatrixXd m);
	static string printMatrixA(MatrixXd m);
	static wstring printMatrixHexW(MatrixXd m);
	static wstring double2hexstrW(double x);
	static string printMatrixHexA(MatrixXd m);
	static string double2hexstrA(double d);
	static double hexastr2doubleA(const string& s);
	static double prod(vector<double> x);
	static vector<double> linspace(double a, double b, size_t N);
	static void Logger(string message);
	static void Logger(wstring message);
	static bool checkMatrix(MatrixXd reference, MatrixXd actual);
	static bool checkMatrix(MatrixXd reference, MatrixXd actual, double precision);
	static bool checkMatrix(MatrixXd expected, MatrixXd actual, double precision, bool print);
	static void WriteToBinary(string fileName, MatrixXd matrix);
	static void WriteToString(string fileName, MatrixXd matrix);
	static bool saveArray(MatrixXd A, const std::string& file_path);
	static MatrixXd ReadBinary(string fileName, int rows, int cols);
	static MatrixXd ReadBinary(string path, string fileName, int rows, int cols);
	//static MatrixXd mult(MatrixXd &a, MatrixXd &b);
	static map<string, double> LoadMock();
	static double innerMock(string key, int thread, int num, int sub);
	static int BinomialCoefficient(int n, int i);
	static int Factorial(int n);
private:
	static map<string, double> mock;
};

