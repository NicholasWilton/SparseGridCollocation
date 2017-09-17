#include "stdafx.h"
#include "EuropeanCallOption.h"
#include "Distributions.h"

using namespace Leicester::Common;

Leicester::SparseGridCollocation::EuropeanCallOption::EuropeanCallOption()
{
}

Leicester::SparseGridCollocation::EuropeanCallOption::EuropeanCallOption(double strike, double maturity)
{
	this->Strike = strike;
	this->Maturity = maturity;
}


Leicester::SparseGridCollocation::EuropeanCallOption::~EuropeanCallOption()
{
}


VectorXd Leicester::SparseGridCollocation::EuropeanCallOption::PayOffFunction(VectorXd S)
{
	VectorXd delta = S.array() - this->Strike;
	VectorXd result = (delta.array() > 0).select(delta, 0);
	return result;
}

MatrixXd Leicester::SparseGridCollocation::EuropeanCallOption::Price(const MatrixXd &X, double r, double sigma)
{
	double strike = this->Strike;
	double maturity = this->Maturity;
	VectorXd t = X.col(0);
	VectorXd S = X.col(1);
	VectorXd M = maturity - t.array();
	int N = X.rows();
	VectorXd P = VectorXd::Ones(N, 1);
	VectorXd d1 = VectorXd::Ones(N, 1);
	VectorXd d2 = VectorXd::Ones(N, 1);

	MatrixXd I0 = (M.array() == 0).select(M, 0);
	MatrixXd I1 = (M.array() != 0).select(M, 0);

	for (int i = 0; i < N; i++)
	{
		if (S(i) -  strike> 0)
			P(i) = S(i) - strike;
		else
			P(i) = 0;

		d1(i) = (log(S(i) / strike) + (r + sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		d2(i) = (log(S(i) / strike) + (r - sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		P(i) = S(i) * Distributions::normCDF(d1(i)) - strike * exp(-r * M(i)) * Distributions::normCDF(d2(i));
	}
	return P;
}