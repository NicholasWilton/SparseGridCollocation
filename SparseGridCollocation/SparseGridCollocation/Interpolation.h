#pragma once
#include "stdafx.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
//using Eigen::Map;
using Eigen::UpLoType;
using Eigen::PartialPivLU;
using namespace std;
//using std::Map;
namespace Leicester
{
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
		MatrixXd getLambda(int id);
		MatrixXd getTX(int id);
		MatrixXd getC(int id);
		MatrixXd getA(int id);
		MatrixXd getU(int id);

		MatrixXd subnumber(int b, int d);
		MatrixXd primeNMatrix(int b, int d);
		VectorXd Replicate(VectorXd v, int totalLength);
		VectorXd Replicate(VectorXd v, int totalLength, int dup);
		void interpolateGeneric(string prefix, double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E,
			vector<string> keys, const map<string, vector<vector<MatrixXd>> > *vInterpolation);
		void interpolateGenericND(string prefix, double coef, double tsec, int b, int d, MatrixXd inx1, MatrixXd inx2, double r, double sigma, double T, double E,
			vector<string> keys, const  map<string, vector<vector<MatrixXd>> > * vInterpolation);
		void shapelambda2DGeneric(string prefix, int threadId, double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
			vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state);
		void shapelambdaNDGeneric(string prefix, int threadId, double coef, double tsec, double r, double sigma, double T, double E, MatrixXd inx1, MatrixXd inx2, MatrixXd N,
			vector<string> keys, const  map<string, vector<vector<MatrixXd>> > * vInterpolation);
	};

}