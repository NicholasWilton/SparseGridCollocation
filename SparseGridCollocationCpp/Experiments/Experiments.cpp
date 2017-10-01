// TestHarness.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "SmoothInitialU.h"
#include "SmoothInitialX.h"
#include "Params.h"
#include "BasketOption.h"
#include <ql/qldefines.hpp>
#ifdef BOOST_MSVC
#  include <ql/auto_link.hpp>
#endif
#include <ql/quantlib.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/mceuropeanengine.hpp>
#include <ql/pricingengines/basket/mceuropeanbasketengine.hpp>
#include <iostream>
#include <boost/timer.hpp>


//using namespace QuantLib;
using namespace Eigen;
using namespace std;
using namespace Leicester;
namespace l = Leicester;
using namespace Leicester::Common;
using namespace Leicester::SparseGridCollocation;
namespace ql = QuantLib;



void Experiment1()
{
	wcout << "Verify MuSiK-c 1D matches MatLab for a European Call Option with 1 asset" << endl;

	Params p;
	p.T = 1.0;
	p.Tdone = 0.0;
	p.Tend = 0.5 * p.T;
	p.dt = 1.0 / 1000.0;
	p.K = 100.0;
	p.r = 0.03;
	p.sigma = 0.15;
	p.theta = 0.5;
	int assets = 1;
	p.inx1 = VectorXd::Zero(assets);
	VectorXd inx2(assets);
	inx2.fill(3.0 * p.K);
	p.inx2 = inx2;
	MatrixXd correlation = MatrixXd::Identity(assets, assets);
	p.useCuda = false;
	p.GenerateSmoothInitialUsing = InitialMethod::MethodOfLines;
	p.SmoothInitialPath = "1\\";

	BasketOption option(100.0, 1, correlation);
	Algorithm* test = new Algorithm();

	vector<MatrixXd> MuSiKc = test->MuSIKc(10, 0, p);

	MatrixXd matLabResults = Utility::ReadBinary("1", "MatLab_MuSIKc_Results_10000_9.dat", 10000, 9);
	MatrixXd matLabMax = Utility::ReadBinary("1", "MatLab_MuSIKc_MAX_9_1.dat", 9, 1);
	MatrixXd matLabRms = Utility::ReadBinary("1", "MatLab_MuSIKc_RMS_9_1.dat", 9, 1);
	MatrixXd matLabTimings = Utility::ReadBinary("1", "MatLab_MuSIKc_Timings_1_13.dat", 1, 13);

	wcout << "MuSIK-c 1D RMS error:" << endl;
	wcout << Utility::printMatrix(MuSiKc[1]) << endl;
	wcout << "MuSIK-c 1D MAX error:" << endl;
	wcout << Utility::printMatrix(MuSiKc[2]) << endl;
	wcout << "MuSIK-c 1D Timings:" << endl;
	MatrixXd timings = MuSiKc[3].transpose();
	wcout << Utility::printMatrix(timings) << endl;
	wcout << "Total=" << timings.sum() << " secs";

	double precision = 0.1;
	wcout << "Checking vs MatLab" << endl;

	wcout << "Individual result elements:" << endl;
	if (Utility::checkMatrix(matLabResults, MuSiKc[0], precision, false))
		wcout << "Information: Individual elements match to precision:" << precision << endl;
	else
		wcout << "Warning: Individual elements did not match" << endl;


	wcout << "RMS error:" << endl;
	if (Utility::checkMatrix(matLabRms, MuSiKc[1], 0.00000001, false))
		wcout << "Success: RMS error elements match to precision:" << precision << endl;
	else
		wcout << "Error: RMS error elements did not match" << endl;


	wcout << "MAX error:" << endl;
	if (Utility::checkMatrix(matLabMax, MuSiKc[2], 0.1, false))
		wcout << "Information: MAX error elements match to precision:" << precision << endl;
	else
		wcout << "Warning: MAX error elements did not match" << endl;

	wcout << "Timining comparison:" << endl;
	if (Utility::checkMatrix(matLabTimings, MuSiKc[3], 0.000001, true))
		wcout << "Information: timings match to precision:" << precision << endl;
	else
		wcout << "Warning: timings did not match" << endl;

	//wcout << getchar() << endl;
}


int main(int argc, char *argv[]) {

	int experiment = 0;
	if (argc < 2)
	{
		wcout << "Useage: Experiments.exe [ExperimentNumber]" << endl;
		wcout << "ExperimentNumber - Specify an experiment 1-10 refer to supporting documentation for details of a particular experiment" << endl;
		return -1;
	}
	else
	{
		experiment = std::atoi(argv[1]);
	}

	experiment = 1;

	if (experiment == 1)
	{
		Experiment1();
	}




	wcout << getchar() << endl;
	return 0;
}
