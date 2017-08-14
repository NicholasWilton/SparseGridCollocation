#pragma once
#include "stdafx.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

class API RBF
{
public:
	RBF();
	~RBF();
	static VectorXd exp(const VectorXd &v);
	static vector<MatrixXd> mqNd(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
	static vector<MatrixXd> mqd2(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
	static vector<MatrixXd> mqd1(const MatrixXd &xx, double x, double c);
};

