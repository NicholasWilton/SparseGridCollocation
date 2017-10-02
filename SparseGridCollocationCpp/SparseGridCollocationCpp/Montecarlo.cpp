#include "stdafx.h"
#include "Montecarlo.h"
#include "Utility.h"


namespace ql = QuantLib;
using namespace std;
//using namespace Leicester::Common;



namespace Leicester
{
	namespace SparseGridCollocation
	{
		Montecarlo::Montecarlo()
		{
		}


		Montecarlo::~Montecarlo()
		{
		}

		SmoothInitial Montecarlo::BasketOption(Params p)
		{
			double Smin = 0;
			double Smax = 3 * p.K;
			double twoPower15Plus1 = pow(2, 15) + 1;

			VectorXd x = VectorXd::LinSpaced(twoPower15Plus1, Smin, Smax);

			VectorXd mcBasket = VectorXd::Zero(x.rows());
			int assets = p.inx1.size();
			std::cout << "Quantlib Montecarlo for Euro Call Basket option (" << assets << " assets)" << endl;

			ql::Calendar calendar = ql::TARGET();
			ql::Date todaysDate(01, ql::Jan, 2000);
			ql::Date settlementDate(01, ql::Jan, 2000);
			ql::Settings::instance().evaluationDate() = todaysDate;
			ql::Option::Type type(ql::Option::Call);
			ql::Real strike = p.K;
			ql::Spread dividendYield = 0.00;
			ql::Rate riskFreeRate = p.r;
			ql::Volatility volatility = p.sigma;
			ql::Date maturity(01, ql::Jan, 2001);
			ql::DayCounter dayCounter = ql::Actual365Fixed();

			for (int i = 0; i < x.rows(); i++)
			{
				boost::timer timer;
				std::cout << "\r" << "underlying price S=" << x[i];

				ql::Real underlying = x[i];

				ql::Handle<ql::Quote> underlyingH(boost::shared_ptr<ql::Quote>(new ql::SimpleQuote(underlying)));

				ql::Handle<ql::YieldTermStructure> flatTermStructure(boost::shared_ptr<ql::YieldTermStructure>(
					new ql::FlatForward(settlementDate, riskFreeRate, dayCounter)));

				ql::Handle<ql::YieldTermStructure> flatDividendTS(boost::shared_ptr<ql::YieldTermStructure>(
					new ql::FlatForward(settlementDate, dividendYield, dayCounter)));

				ql::Handle<ql::BlackVolTermStructure> flatVolTS(boost::shared_ptr<ql::BlackVolTermStructure>(
					new ql::BlackConstantVol(settlementDate, calendar, volatility,
						dayCounter)));

				boost::shared_ptr<ql::PlainVanillaPayoff> payoff(new ql::PlainVanillaPayoff(type, strike));

				boost::shared_ptr<ql::BasketPayoff> basketPayoff(new ql::AverageBasketPayoff(payoff, assets));

				boost::shared_ptr<ql::BlackScholesMertonProcess> bsmProcess(new ql::BlackScholesMertonProcess(underlyingH, flatDividendTS,
					flatTermStructure, flatVolTS));

				boost::shared_ptr<ql::Exercise> exercise(new ql::EuropeanExercise(maturity));
				ql::BasketOption basketOption(basketPayoff, exercise);

				ql::Size timeSteps;
				timeSteps = 1;
				
				ql::Size mcSeed = 42;
				std::vector<boost::shared_ptr<ql::StochasticProcess1D> > procs;
				procs.push_back(bsmProcess);

				ql::Matrix correlationMatrix(1, 1, 0);
				for (ql::Integer j = 0; j < 1; j++) {
					correlationMatrix[j][j] = 1.0;
				}

				boost::shared_ptr<ql::StochasticProcessArray> processArray(new ql::StochasticProcessArray(procs, correlationMatrix));

				boost::shared_ptr<ql::PricingEngine> mcengine1;
				mcengine1 = ql::MakeMCEuropeanBasketEngine<ql::PseudoRandom, ql::GaussianStatistics>(processArray)
					.withStepsPerYear(1)
					.withSamples(10000)
					.withSeed(42);

				boost::shared_ptr<ql::PricingEngine> mcengine2;
				mcengine2 = ql::MakeMCEuropeanEngine<ql::PseudoRandom>(bsmProcess)
					.withStepsPerYear(1)
					.withSamples(10000)
					.withSeed(42);


				basketOption.setPricingEngine(mcengine1);
				ql::Real u = basketOption.NPV();
				mcBasket[i] = (double)u;
			}

			//Utility::saveMatrix(mcBasket, "MCBasket.txt");
			SmoothInitial result;
			result.S = x;
			result.U = mcBasket;
			result.T = 0.8; //warning this is arbitrarly hard coded.
			return result;
		}
		
	}
}
