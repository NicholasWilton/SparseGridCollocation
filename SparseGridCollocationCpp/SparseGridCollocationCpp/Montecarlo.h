#pragma once

#include <ql/qldefines.hpp>
#ifdef BOOST_MSVC
#  include <ql/auto_link.hpp>
#endif
#include <ql/quantlib.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/mceuropeanengine.hpp>
#include <ql/pricingengines/basket/mceuropeanbasketengine.hpp>
#include <boost/timer.hpp>
#include "Params.h"
#include "SmoothInitial.h"

using namespace Eigen;
using namespace std;

namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API Montecarlo
		{
		public:
			Montecarlo();
			~Montecarlo();
			static SmoothInitial BasketOption(Params p);

		};
	}
}
