#pragma once

using namespace Eigen;
using namespace std;

class API EuropeanCallOption
{
public:
	EuropeanCallOption();
	~EuropeanCallOption();
	static VectorXd PayOffFunction(VectorXd S, double K);
	static MatrixXd Price(const MatrixXd &X, double r, double sigma, double T, double E);
};

