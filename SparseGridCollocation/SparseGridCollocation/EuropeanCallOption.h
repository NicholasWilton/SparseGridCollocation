#pragma once
#include "Option.h"

using namespace Eigen;
using namespace std;

namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API EuropeanCallOption : public Option
		{
		public:
			EuropeanCallOption();
			EuropeanCallOption(double strike, double maturity);
			~EuropeanCallOption();
			VectorXd PayOffFunction(VectorXd S);
			MatrixXd Price(const MatrixXd &X, double r, double sigma);
		private:
			double Strike;
			double Maturity;
		};
	}
}