#pragma once
using namespace Eigen;
namespace Leicester
{
	class API Params
	{
	public:
		Params();
		~Params();
		double T;
		double Tdone;
		double Tend;
		double dt;
		double K;
		double r;
		double sigma;
		double theta;
		VectorXd inx1;
		VectorXd inx2;
	};
}

