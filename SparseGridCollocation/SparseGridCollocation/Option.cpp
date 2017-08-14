#include "stdafx.h"
#include "Option.h"

Option::Option()
{
}

Option::Option(double strike, double maturity)
{
	this->Strike = strike;
	this->Maturity = maturity;
}


Option::~Option()
{
}


VectorXd Option::PayOffFunction(VectorXd S) { return VectorXd(); }
MatrixXd Option::Price(const MatrixXd &X, double r, double sigma) { return MatrixXd(); }