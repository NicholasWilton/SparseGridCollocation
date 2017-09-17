#include "stdafx.h"
#include "Distributions.h"


double Leicester::Common::Distributions::normCDF(double value)
{
	return 0.5 * erfc(-value * (1 / sqrt(2)));
}
