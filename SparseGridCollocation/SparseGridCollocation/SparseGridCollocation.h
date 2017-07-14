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

	
	MatrixXd ECP(MatrixXd X, double r, double sigma, double T, double E);
	double inner_test(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);

	double normCDF(double value);
	double RootMeanSquare(VectorXd v);

	vector<VectorXd> MuSIKGeneric(int level);
	map<string, vector<vector<MatrixXd>>> GetInterpolation();

private:
	map<string, vector<vector<MatrixXd>>> vInterpolation;
	//key - index from matlab code eg lamb3, TX3 etc -> '3'
	//value - vector of items lamb, TX, C, A
	//		- each in turn is a vector of Matrices

	vector<MatrixXd> vLamb;
	vector<MatrixXd> vTX;
	vector<MatrixXd> vC;
	vector<MatrixXd> vA;
	


};