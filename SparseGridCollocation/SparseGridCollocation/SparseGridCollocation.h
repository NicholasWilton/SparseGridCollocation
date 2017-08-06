#pragma once

#include "stdafx.h"
#include "Params.h"

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

	
	MatrixXd ECP(const MatrixXd &X, double r, double sigma, double T, double E);
	
	double normCDF(double value);
	double RootMeanSquare(VectorXd v);

	vector<MatrixXd> MuSIKc(int upper, int lower, Params p);
	vector<MatrixXd> MuSIKc(int upper, int lower, Params p, map<string, vector<vector<MatrixXd>>>& interpolation);
	map<string, vector<vector<MatrixXd>>> GetInterpolationState();
	map<int, MatrixXd> GetU();
	
	vector<VectorXd> MethodOfLines(Params p);
	vector<VectorXd> MethodOfLines(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2);
	vector<VectorXd> EuroCallOption1D(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2);
	
	VectorXd Select(VectorXd A, double notEqual);

private:
	map<string, vector<vector<MatrixXd>>> vInterpolation;
	map<int, MatrixXd> uMatrix;
	//key - index from matlab code eg lamb3, TX3 etc -> '3'
	//value - vector of items lamb, TX, C, A
	//		- each in turn is a vector of Matrices

	vector<MatrixXd> vLamb;
	vector<MatrixXd> vTX;
	vector<MatrixXd> vC;
	vector<MatrixXd> vA;
	


};