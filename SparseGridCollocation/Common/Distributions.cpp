#include "stdafx.h"
#include "Distributions.h"


Distributions::Distributions()
{
}


Distributions::~Distributions()
{
}


double Distributions::normCDF(double value)
{
	return 0.5 * erfc(-value * (1 / sqrt(2)));
}
