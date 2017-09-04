#include "stdafx.h"
#include "BasketOption.h"
#include "Distributions.h"
#include "TestNodes.h"


Leicester::BasketOption::BasketOption()
{
}

Leicester::BasketOption::BasketOption(double strike, double maturity, MatrixXd correlation)
{
	this->Strike = strike;
	this->Maturity = maturity;
	this->Underlying = correlation.rows();
	this->correlation = correlation;
}


Leicester::BasketOption::~BasketOption()
{
}


VectorXd Leicester::BasketOption::PayOffFunction(MatrixXd S)
{

	VectorXd delta = S.rowwise().mean().array() - this->Strike;
	
	VectorXd result = (delta.array() > 0).select(delta, 0);
	return result;
}

VectorXd Leicester::BasketOption::PayOffS(MatrixXd S)
{
	return S.rowwise().mean().array();
}

MatrixXd Leicester::BasketOption::NodesAroundStrike(const MatrixXd &X, double strike, double radius)
{
	VectorXd avg = X.rowwise().sum() / X.cols();

	double range = strike * radius;

	MatrixXd result(X.rows(), X.cols());
	result.fill(0);
	vector<int> nonzero;
	for (int row =0 ; row < avg.size(); row++)
	{
		if (avg[row] >= strike - range & avg[row] <= strike + range)
		{
			result.row(row) = X.row(row);
			nonzero.push_back(row);
		}
	}
	MatrixXd selected(nonzero.size(), X.cols());
	int count = 0;
	for (auto i : nonzero)
	{
		selected.row(count) = result.row(i);
		count++;
	}
	return selected;
}

MatrixXd Leicester::BasketOption::NodesAroundStrikeFromGrid(const MatrixXd &X, double strike, double radius)
{
	double range = strike * radius;
	MatrixXd lower(X.rows(), X.cols());
	lower.fill(strike - range);
	MatrixXd upper(X.rows(), X.cols());
	upper.fill(strike + range);
	MatrixXd zero = MatrixXd::Zero(X.rows(), X.cols());

	MatrixXd X1 = (X.array() >= lower.array()).select(X, zero);
	MatrixXd X2 = (X.array() <= upper.array()).select(X1, zero);

	vector<int> nonzero;
	for (int row = 0; row < X2.rows(); row++)
	{
		if (X2.row(row).sum() != 0)
			nonzero.push_back(row);
	}

	MatrixXd selected(nonzero.size(), X.cols());
	int count = 0;
	for (auto i : nonzero)
	{
		selected.row(count) = X2.row(i);
		count++;
	}

	MatrixXd result = TestNodes::CartesianProduct(selected);
	return result;
}

MatrixXd Leicester::BasketOption::Price(const MatrixXd &X, double r, double sigma)
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
		if (S(i) - strike> 0)
			P(i) = S(i) - strike;
		else
			P(i) = 0;

		d1(i) = (log(S(i) / strike) + (r + sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		d2(i) = (log(S(i) / strike) + (r - sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		P(i) = S(i) * Distributions::normCDF(d1(i)) - strike * exp(-r * M(i)) * Distributions::normCDF(d2(i));
	}
	return P;
}