#pragma once
#include "stdafx.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
//using Eigen::Map;
using Eigen::UpLoType;
using Eigen::PartialPivLU;
using namespace std;
//using std::Map;

class API Interpolation
{
	vector<vector<MatrixXd>> result;
	int count = 0;
	map<int, MatrixXd> Lambda;
	map<int, MatrixXd> TX;
	map<int, MatrixXd> C;
	map<int, MatrixXd> A;
	map<int, MatrixXd> U;

	
	
public:
	//Interpolation();
	//Interpolation(const Interpolation & obj);
	//~Interpolation();
	vector<vector<MatrixXd>> getResult();
	MatrixXd subnumber(int b, int d);
	MatrixXd primeNMatrix(int b, int d);
	void interpolateGeneric(string prefix, double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E, 
		vector<string> keys, const  map<string, vector<vector<MatrixXd>> > * vInterpolation);
	void shapelambda2DGeneric(int threadId, double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N, 
		vector<string> keys, const  map<string, vector<vector<MatrixXd>> > * vInterpolation);
};

