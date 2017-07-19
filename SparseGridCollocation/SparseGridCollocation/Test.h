#pragma once
#include "stdafx.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Matrix2d;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using Eigen::Map;
using Eigen::aligned_allocator;
using namespace std;

class API Test
{
public:
	Test();
	~Test();

	static double inner(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);
	static double innerX(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);
	static double innerY(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);
	static double innerZ(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);
	static MatrixXd mult(MatrixXd &a, MatrixXd &b);
	static VectorXd inter(MatrixXd X, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);
	static MatrixXd MatrixExp(MatrixXd m);
	static double MatLabRounding(double d);
	static MatrixXd MatLabRounding(MatrixXd m);
	
	static map<string, double> LoadMock();
	static double innerMock(string key, int thread, int num, int sub);
private :
	static map<string, double> mock;
};

struct Data {
	string key;
	int thread;
	int num;
	int sub;
	double value;
};

