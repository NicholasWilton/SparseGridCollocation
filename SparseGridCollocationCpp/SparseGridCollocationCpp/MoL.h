#pragma once

#include "Params.h";
#include "SmoothInitial.h"

using namespace Eigen;
using namespace std;

namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API MoL
		{
		public:
			MoL();
			~MoL();
			static vector<VectorXd> MethodOfLines(Params p);
			static vector<VectorXd> MethodOfLines(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2);
			static vector<VectorXd> EuroCallOption1D(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2);
			static SmoothInitial EuroCallOptionND(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, MatrixXd correlation);
			static vector<vector<VectorXd>> EuroCallOptionND_ODE(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation);
			static SmoothInitial MethodOfLinesND(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation);
			static vector<vector<VectorXd>> MethodOfLinesND_ODE(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation);
			static SmoothInitial MethodOfLinesND(Params p, MatrixXd correlation);
			static MatrixXd PushAndQueueBoundaries(MatrixXd A, VectorXd inx2, double r, double Tdone, double K);
			static vector<MatrixXd> SetupBasket(int assets, int testNodeNumber, int centralNodeNumber, double aroundStrikeRange, double strike);
			static MatrixXd TakeMeans(MatrixXd M);
		};
	}
}
