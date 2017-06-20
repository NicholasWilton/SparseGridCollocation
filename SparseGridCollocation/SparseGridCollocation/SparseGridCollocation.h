#pragma once

#define API _declspec(dllexport)


#include "stdafx.h"
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using Eigen::Map;
using namespace std;

class API SparseGridCollocation
{
public:
	//SparseGridCollocation(void);
	static wstring printMatrix(MatrixXd *matrix);
	static wstring printMatrixI(MatrixXd m);
	vector<MatrixXd> mqd2(MatrixXd TP, MatrixXd CN, MatrixXd A, MatrixXd C);
	void shapelambda2D(MatrixXd *lamb, MatrixXd *TX, MatrixXd *C, MatrixXd *A, double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N);

};