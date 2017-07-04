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
	static vector<MatrixXd> mqd2(MatrixXd TP, MatrixXd CN, MatrixXd A, MatrixXd C);
};

