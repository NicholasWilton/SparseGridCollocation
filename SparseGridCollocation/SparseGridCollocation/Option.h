#pragma once
using namespace Eigen;
using namespace std;
namespace Leicester
{
	class API Option
	{
	public:

		Option();
		Option(double strike, double maturity);
		~Option();

		virtual VectorXd PayOffFunction(VectorXd S);
		virtual MatrixXd Price(const MatrixXd &X, double r, double sigma);
		double TDone;
	private:
		double Strike;
		double Maturity;
	};
}