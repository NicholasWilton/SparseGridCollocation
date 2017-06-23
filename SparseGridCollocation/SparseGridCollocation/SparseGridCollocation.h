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

class API SparseGridCollocation
{
public:
	SparseGridCollocation();
	MatrixXd subnumber(int b, int d);
	MatrixXd primeNMatrix(int b, int d);
	vector<MatrixXd> mqd2(MatrixXd TP, MatrixXd CN, MatrixXd A, MatrixXd C);
	MatrixXd PDE(MatrixXd node, double r, double sigma, 
		vector<MatrixXd> lambda2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2, 
		vector<MatrixXd> lambda3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3 );
	
	vector<MatrixXd> shapelambda2D(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N);
	vector<MatrixXd> shapelambda2D_1(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
		vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2, vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3);

	vector<vector<MatrixXd>> interpolate(double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E);
	vector<vector<MatrixXd>> interpolate1(double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E,
		vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2, vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3);

	void MuSIK();
private:
	vector<vector<MatrixXd>> vInterpolation;
	vector<MatrixXd> vLamb;
	vector<MatrixXd> vTX;
	vector<MatrixXd> vC;
	vector<MatrixXd> vA;
	


};