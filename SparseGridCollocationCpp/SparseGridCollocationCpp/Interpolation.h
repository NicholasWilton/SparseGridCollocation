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
	namespace SparseGridCollocation
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
			static int callCount;
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
				vector<string> keys, const  map<string, vector<vector<MatrixXd>> > * vInterpolation, bool useCuda);
			
			void shapelambda2DGeneric(string prefix, int threadId, double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
				vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state, MatrixXd TP);
			void ShapeLambda2D(std::vector<Eigen::MatrixXd> &mqd, double sigma, double &r, double num, std::vector<std::string> &keys, Eigen::MatrixXd &TP, std::map<std::string, std::vector<std::vector<Eigen::MatrixXd>>> &vInterpolation, double inx1, double inx2, double E, double T, double tsec, int &threadId, Eigen::MatrixXd &c, Eigen::MatrixXd &a);
			void shapelambdaNDGeneric(string prefix, int threadId, double coef, double tsec, double r, double sigma, double T, double E, MatrixXd inx1, MatrixXd inx2, MatrixXd N,
				vector<string> keys, const  map<string, vector<vector<MatrixXd>> > * vInterpolation);
			

			MatrixXd GenerateNodes(double coef, double tsec, double inx1, double inx2, MatrixXd N);
		};
	}
}