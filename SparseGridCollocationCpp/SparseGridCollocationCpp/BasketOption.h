#pragma once
#include "Option.h"

using namespace Eigen;
using namespace std;
namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API BasketOption : public Option
		{
		public:
			BasketOption();
			BasketOption(double strike, double maturity, MatrixXd correlation);
			~BasketOption();
			VectorXd PayOffFunction(MatrixXd S);
			VectorXd PayOffS(MatrixXd S);
			virtual MatrixXd Price(const MatrixXd &X, double r, double sigma);
			static MatrixXd NodesAroundStrike(const MatrixXd &X, double strike, double radius);
			static MatrixXd NodesAroundStrikeFromGrid(const MatrixXd &X, double strike, double radius);
			double Strike;
			double Maturity;
			int Underlying;
			MatrixXd correlation;

		};
	}
}
