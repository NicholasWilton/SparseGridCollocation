#pragma once
#include "stdafx.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace std;

class API InterTest
{
private:
	map<string, VectorXd> *result = new map<string, VectorXd>();
	
public:
	InterTest();
	~InterTest();
	void Execute(map<string, vector<vector<MatrixXd>> > vInterpolation, MatrixXd TX);
	void parallel(string id, const MatrixXd &X, const vector<MatrixXd> &lamb, const vector<MatrixXd> &TX, const vector<MatrixXd> &C, const vector<MatrixXd> &A);
	VectorXd serial(MatrixXd X, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A);
	VectorXd GetResult(string id);
};

