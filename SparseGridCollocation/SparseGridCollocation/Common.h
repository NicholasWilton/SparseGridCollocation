#pragma once

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
};

