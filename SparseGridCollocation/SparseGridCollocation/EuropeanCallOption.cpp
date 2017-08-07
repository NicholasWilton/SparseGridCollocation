#include "stdafx.h"
#include "EuropeanCallOption.h"
#include "Distributions.h"


EuropeanCallOption::EuropeanCallOption()
{
}


EuropeanCallOption::~EuropeanCallOption()
{
}


VectorXd EuropeanCallOption::PayOffFunction(VectorXd S, double K)
{
	VectorXd delta = S.array() - K;
	VectorXd result = (delta.array() > 0).select(delta, 0);
	return result;
}

MatrixXd EuropeanCallOption::Price(const MatrixXd &X, double r, double sigma, double T, double E)
{
	VectorXd t = X.col(0);
	VectorXd S = X.col(1);
	VectorXd M = T - t.array();
	int N = X.rows();
	VectorXd P = VectorXd::Ones(N, 1);
	VectorXd d1 = VectorXd::Ones(N, 1);
	VectorXd d2 = VectorXd::Ones(N, 1);

	MatrixXd I0 = (M.array() == 0).select(M, 0);
	MatrixXd I1 = (M.array() != 0).select(M, 0);

	for (int i = 0; i < N; i++)
	{
		if (S(i) - E > 0)
			P(i) = S(i) - E;
		else
			P(i) = 0;

		d1(i) = (log(S(i) / E) + (r + sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		d2(i) = (log(S(i) / E) + (r - sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		P(i) = S(i) * Distributions::normCDF(d1(i)) - E * exp(-r * M(i)) * Distributions::normCDF(d2(i));
	}
	return P;
}