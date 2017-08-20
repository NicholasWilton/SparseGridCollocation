#include "stdafx.h"
#include "Option.h"

Leicester::Option::Option()
{
}

Leicester::Option::Option(double strike, double maturity)
{
	this->Strike = strike;
	this->Maturity = maturity;
}


Leicester::Option::~Option()
{
}


VectorXd Leicester::Option::PayOffFunction(VectorXd S) { return VectorXd(); }
MatrixXd Leicester::Option::Price(const MatrixXd &X, double r, double sigma) { return MatrixXd(); }