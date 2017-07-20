#pragma once
#include "stdafx.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

class API PDE
{
public:
	PDE();
	~PDE();
	static MatrixXd BlackScholes(const MatrixXd &node, double r, double sigma,
		const vector<MatrixXd> &lambda2, const vector<MatrixXd> &TX2, const vector<MatrixXd> &C2, const vector<MatrixXd> &A2,
		const vector<MatrixXd> &lambda3, const vector<MatrixXd> &TX3, const vector<MatrixXd> &C3, const vector<MatrixXd> A3);
};

