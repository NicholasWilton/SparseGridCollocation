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
		BasketOption(double strike, double maturity, MatrixXd correlation);
		~BasketOption();
		VectorXd PayOffFunction(MatrixXd S);
		virtual MatrixXd Price(const MatrixXd &X, double r, double sigma);
		static MatrixXd NodesAroundStrike(const MatrixXd &X, double strike, double radius);
		double Strike;
		double Maturity;
		int Underlying;
		MatrixXd correlation;

	private:

	};
}
