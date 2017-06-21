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
	//SparseGridCollocation(void);
	//static wstring printMatrix(MatrixXd *matrix);
	//static wstring printMatrixI(MatrixXd m);
	vector<MatrixXd> mqd2(MatrixXd TP, MatrixXd CN, MatrixXd A, MatrixXd C);
	vector<MatrixXd> shapelambda2D(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N);

};