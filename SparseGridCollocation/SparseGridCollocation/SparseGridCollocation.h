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
	double inner_test(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);
	
	vector<MatrixXd> shapelambda2D(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N);
	vector<MatrixXd> shapelambda2D_1(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
		vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2, vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3);
	vector<MatrixXd> shapelambda2D_2(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
		vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2,
		vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3,
		vector<MatrixXd> lamb_3, vector<MatrixXd> TX_3, vector<MatrixXd> C_3, vector<MatrixXd> A_3,
		vector<MatrixXd> lamb4, vector<MatrixXd> TX4, vector<MatrixXd> C4, vector<MatrixXd> A4);

	void interpolateGeneric(string prefix, double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E, vector<string> keys);
	vector<MatrixXd> shapelambda2DGeneric(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N, vector<string> keys);

	vector<vector<MatrixXd>> interpolate(double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E);
	vector<vector<MatrixXd>> interpolate1(double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E,
		vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2, vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3);
	vector<vector<MatrixXd>> interpolate2(double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E,
		vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2,
		vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3,
		vector<MatrixXd> lamb_3, vector<MatrixXd> TX_3, vector<MatrixXd> C_3, vector<MatrixXd> A_3,
		vector<MatrixXd> lamb4, vector<MatrixXd> TX4, vector<MatrixXd> C4, vector<MatrixXd> A4);

	void MuSIK();
	void MuSIKGeneric();

private:
	map<string, vector<vector<MatrixXd>> > vInterpolation;
	//key - index from matlab code eg lamb3, TX3 etc -> '3'
	//value - vector of items lamb, TX, C, A
	//		- each in turn is a vector of Matrices

	vector<MatrixXd> vLamb;
	vector<MatrixXd> vTX;
	vector<MatrixXd> vC;
	vector<MatrixXd> vA;
	


};