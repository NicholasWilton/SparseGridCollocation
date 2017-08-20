#pragma once
#include "Option.h"

using namespace Eigen;
using namespace std;
namespace Leicester
{
	class API BasketOption : public Option
	{
	public:
		BasketOption();
		BasketOption(double strike, double maturity, int underlying);
		~BasketOption();
		VectorXd PayOffFunction(MatrixXd S);
		virtual MatrixXd Price(const MatrixXd &X, double r, double sigma);
		double Strike;
		double Maturity;
		int Underlying;

	private:

	};
}
