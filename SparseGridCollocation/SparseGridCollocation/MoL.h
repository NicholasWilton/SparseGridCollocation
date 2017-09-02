#pragma once

#include "Params.h";

using namespace Eigen;
using namespace std;
namespace Leicester
{
	class API MoL
	{
	public:
		MoL();
		~MoL();
		static vector<VectorXd> MethodOfLines(Params p);
		static vector<VectorXd> MethodOfLines(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2);
		static vector<VectorXd> EuroCallOption1D(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2);
		static vector<vector<VectorXd>> EuroCallOptionND(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation);
		static vector<vector<VectorXd>> EuroCallOptionND_ODE(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation);
		static vector<vector<VectorXd>> MethodOfLinesND(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation);
		static vector<vector<VectorXd>> MethodOfLinesND_ODE(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation);
		static vector<MatrixXd> MethodOfLinesND(Params p, MatrixXd correlation);
		static MatrixXd PushAndQueueBoundaries(MatrixXd A, VectorXd inx2, double r, double Tdone, double K);
	};
}
