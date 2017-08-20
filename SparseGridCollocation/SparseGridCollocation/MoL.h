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
	};
}
