#include "stdafx.h"
#include "Option.h"

Leicester::SparseGridCollocation::Option::Option()
{
}

Leicester::SparseGridCollocation::Option::Option(double strike, double maturity)
{
	this->Strike = strike;
	this->Maturity = maturity;
}


Leicester::SparseGridCollocation::Option::~Option()
{
}


VectorXd Leicester::SparseGridCollocation::Option::PayOffFunction(VectorXd S) { return VectorXd(); }
MatrixXd Leicester::SparseGridCollocation::Option::Price(const MatrixXd &X, double r, double sigma) { return MatrixXd(); }