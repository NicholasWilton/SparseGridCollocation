// TestHarness.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "SmoothInitialU.h"
#include "SmoothInitialX.h"
#include "Params.h"
#include "BasketOption.h"
#include "Mol.h"
//#include "Montecarlo.h"
//#include <ql/qldefines.hpp>
//#ifdef BOOST_MSVC
//#  include <ql/auto_link.hpp>
//#endif
//#include <ql/quantlib.hpp>
//#include <ql/instruments/vanillaoption.hpp>
//#include <ql/pricingengines/vanilla/mceuropeanengine.hpp>
//#include <ql/pricingengines/basket/mceuropeanbasketengine.hpp>
//#include <iostream>
//#include <boost/timer.hpp>


//using namespace QuantLib;
using namespace Eigen;
using namespace std;
using namespace Leicester;
namespace l = Leicester;
using namespace Leicester::Common;
using namespace Leicester::SparseGridCollocation;
//namespace ql = QuantLib;


void Experiment2()
{
	wcout << "C++ CUDA Experiment 2" << endl << endl;
	wcout << "Verify SiK-c 1D matches MatLab for a European Call Option with 1 asset" << endl << endl;

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
	p.useCuda = true;
	p.GenerateSmoothInitialUsing = InitialMethod::MethodOfLines;
	p.SmoothInitialPath = "2\\";

	BasketOption option(100.0, 1, correlation);
	Algorithm* test = new Algorithm();

	vector<MatrixXd> SiKc = test->SIKc(10, 0, p);

	MatrixXd matLabResults = Utility::ReadBinary("2", "MatLab_SIKc_Results_10000_9.dat", 10000, 9);
	MatrixXd matLabMax = Utility::ReadBinary("2", "MatLab_SIKc_MAX_9_1.dat", 9, 1);
	MatrixXd matLabRms = Utility::ReadBinary("2", "MatLab_SIKc_RMS_9_1.dat", 9, 1);
	MatrixXd matLabTimings = Utility::ReadBinary("2", "MatLab_SIKc_Timings_1_10.dat", 1, 13);

	wcout << "SIK-c 1D RMS error:" << endl;
	wcout << Utility::printMatrix(SiKc[1]) << endl;
	wcout << "SIK-c 1D MAX error:" << endl;
	wcout << Utility::printMatrix(SiKc[2]) << endl;
	wcout << "SIK-c 1D Timings:" << endl;
	MatrixXd timings = SiKc[3].transpose();
	wcout << Utility::printMatrix(timings) << endl;
	wcout << "Total=" << timings.sum() << " secs";

	double precision = 0.1;
	wcout << "Checking vs MatLab" << endl;

	wcout << "Individual result elements:" << endl;
	if (Utility::checkMatrix(matLabResults, SiKc[0], precision, false))
		wcout << "Information: Individual elements match to precision:" << precision << endl;
	else
		wcout << "Warning: Individual elements did not match" << endl;


	wcout << "RMS error:" << endl;
	if (Utility::checkMatrix(matLabRms, SiKc[1], 0.00000001, false))
		wcout << "Success: RMS error elements match to precision:" << precision << endl;
	else
		wcout << "Error: RMS error elements did not match" << endl;


	wcout << "MAX error:" << endl;
	if (Utility::checkMatrix(matLabMax, SiKc[2], 0.1, false))
		wcout << "Information: MAX error elements match to precision:" << precision << endl;
	else
		wcout << "Warning: MAX error elements did not match" << endl;

	wcout << "Timining comparison:" << endl;
	if (Utility::checkMatrix(matLabTimings, SiKc[3], 0.000001, true))
		wcout << "Information: timings match to precision:" << precision << endl;
	else
		wcout << "Warning: timings did not match" << endl;

	Utility::WriteToTxt(matLabResults, "2\\matLab.Results.txt");
	Utility::WriteToTxt(matLabRms, "2\\matLab.Rms.txt");
	Utility::WriteToTxt(matLabMax, "2\\matLab.Max.txt");
	Utility::WriteToTxt(matLabTimings, "2\\matLab.Timings.txt");

	Utility::WriteToTxt(SiKc[0], "2\\cu.Results.txt");
	Utility::WriteToTxt(SiKc[1], "2\\cu.Rms.txt");
	Utility::WriteToTxt(SiKc[2], "2\\cu.Max.txt");
	Utility::WriteToTxt(SiKc[3], "2\\cu.Timings.txt");

	//wcout << getchar() << endl;
}

void Experiment3()
{
	wcout << "C++ CUDA Experiment 3" << endl << endl;
	wcout << "Verify MuSiK-c 1D matches MatLab for a European Call Option with 1 asset" << endl << endl;

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
	p.useCuda = true;
	p.GenerateSmoothInitialUsing = InitialMethod::MethodOfLines;
	p.SmoothInitialPath = "3\\";

	BasketOption option(100.0, 1, correlation);
	Algorithm* test = new Algorithm();

	vector<MatrixXd> MuSiKc = test->MuSIKc(10, 0, p);

	MatrixXd matLabResults = Utility::ReadBinary("3", "MatLab_MuSIKc_Results_10000_9.dat", 10000, 9);
	MatrixXd matLabMax = Utility::ReadBinary("3", "MatLab_MuSIKc_MAX_9_1.dat", 9, 1);
	MatrixXd matLabRms = Utility::ReadBinary("3", "MatLab_MuSIKc_RMS_9_1.dat", 9, 1);
	MatrixXd matLabTimings = Utility::ReadBinary("3", "MatLab_MuSIKc_Timings_1_10.dat", 1, 13);

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

	Utility::WriteToTxt(matLabResults, "3\\matLab.Results.txt");
	Utility::WriteToTxt(matLabRms, "3\\matLab.Rms.txt");
	Utility::WriteToTxt(matLabMax, "3\\matLab.Max.txt");
	Utility::WriteToTxt(matLabTimings, "3\\matLab.Timings.txt");

	Utility::WriteToTxt(MuSiKc[0], "3\\cu.Results.txt");
	Utility::WriteToTxt(MuSiKc[1], "3\\cu.Rms.txt");
	Utility::WriteToTxt(MuSiKc[2], "3\\cu.Max.txt");
	Utility::WriteToTxt(MuSiKc[3], "3\\cu.Timings.txt");

	//wcout << getchar() << endl;
}

void Experiment5()
{
	wcout << "C++ CUDA Experiment 5" << endl << endl;
	wcout << "Verify MuSiK-c N-Dimensions with a Basket Option with 1 asset matches MatLab for a European Call Option with 1 asset" << endl << endl;

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
	p.useCuda = true;
	p.GenerateSmoothInitialUsing = InitialMethod::MethodOfLines;
	p.SmoothInitialPath = "5\\";

	BasketOption option(100.0, 1, correlation);
	Algorithm* test = new Algorithm();

	vector<MatrixXd> MuSiKcND = test->MuSIKcND(10, 0, option, p);

	MatrixXd matLabResults = Utility::ReadBinary("5", "MatLab_MuSIKc_Results_10000_9.dat", 10000, 9);
	MatrixXd matLabMax = Utility::ReadBinary("5", "MatLab_MuSIKc_MAX_9_1.dat", 9, 1);
	MatrixXd matLabRms = Utility::ReadBinary("5", "MatLab_MuSIKc_RMS_9_1.dat", 9, 1);
	MatrixXd matLabTimings = Utility::ReadBinary("5", "MatLab_MuSIKc_Timings_1_10.dat", 1, 13);

	wcout << "MuSIK-c ND RMS error:" << endl;
	wcout << Utility::printMatrix(MuSiKcND[1]) << endl;
	wcout << "MuSIK-c ND MAX error:" << endl;
	wcout << Utility::printMatrix(MuSiKcND[2]) << endl;
	wcout << "MuSIK-c ND Timings:" << endl;
	MatrixXd timings = MuSiKcND[3].transpose();
	wcout << Utility::printMatrix(timings) << endl;
	wcout << "Total=" << timings.sum() << " secs";

	double precision = 0.1;
	wcout << "Checking vs MatLab" << endl;

	wcout << "Individual result elements:" << endl;
	if (Utility::checkMatrix(matLabResults, MuSiKcND[0], precision, false))
		wcout << "Information: Individual elements match to precision:" << precision << endl;
	else
		wcout << "Warning: Individual elements did not match" << endl;


	wcout << "RMS error:" << endl;
	if (Utility::checkMatrix(matLabRms, MuSiKcND[1], 0.00000001, false))
		wcout << "Success: RMS error elements match to precision:" << precision << endl;
	else
		wcout << "Error: RMS error elements did not match" << endl;


	wcout << "MAX error:" << endl;
	if (Utility::checkMatrix(matLabMax, MuSiKcND[2], 0.1, false))
		wcout << "Information: MAX error elements match to precision:" << precision << endl;
	else
		wcout << "Warning: MAX error elements did not match" << endl;

	wcout << "Timining comparison:" << endl;
	if (Utility::checkMatrix(matLabTimings, MuSiKcND[3], 0.000001, true))
		wcout << "Information: timings match to precision:" << precision << endl;
	else
		wcout << "Warning: timings did not match" << endl;

	Utility::WriteToTxt(matLabResults, "5\\matLab.Results.txt");
	Utility::WriteToTxt(matLabRms, "5\\matLab.Rms.txt");
	Utility::WriteToTxt(matLabMax, "5\\matLab.Max.txt");
	Utility::WriteToTxt(matLabTimings, "5\\matLab.Timings.txt");

	Utility::WriteToTxt(MuSiKcND[0], "5\\cu.Results.txt");
	Utility::WriteToTxt(MuSiKcND[1], "5\\cu.Rms.txt");
	Utility::WriteToTxt(MuSiKcND[2], "5\\cu.Max.txt");
	Utility::WriteToTxt(MuSiKcND[3], "5\\cu.Timings.txt");
	//wcout << getchar() << endl;
}

void Experiment7()
{
	wcout << "C++ CUDA Experiment 7" << endl << endl;
	wcout << "Do MuSiK-c N-Dimensions with a Basket Option with 2 assets and Quantlib MC as method of choice" << endl << endl;

	Params p;
	p.T = 1.0;
	p.Tdone = 0.0;
	p.Tend = 0.5 * p.T;
	p.dt = 1.0 / 1000.0;
	p.K = 100.0;
	p.r = 0.03;
	p.sigma = 0.15;
	p.theta = 0.5;
	int assets = 2;
	p.inx1 = VectorXd::Zero(assets);
	VectorXd inx2(assets);
	inx2.fill(3.0 * p.K);
	p.inx2 = inx2;
	MatrixXd correlation = MatrixXd::Identity(assets, assets);
	p.useCuda = true;
	p.GenerateSmoothInitialUsing = InitialMethod::MonteCarlo;
	p.SmoothInitialPath = "7\\";

	BasketOption option(100.0, 1, correlation);
	Algorithm* test = new Algorithm();

	vector<MatrixXd> MuSiKcND = test->MuSIKcND(6, 0, option, p);


	wcout << "MuSIK-c ND RMS error:" << endl;
	wcout << Utility::printMatrix(MuSiKcND[1]) << endl;
	wcout << "MuSIK-c ND MAX error:" << endl;
	wcout << Utility::printMatrix(MuSiKcND[2]) << endl;
	wcout << "MuSIK-c ND Timings:" << endl;
	MatrixXd timings = MuSiKcND[3].transpose();
	wcout << Utility::printMatrix(timings) << endl;
	wcout << "Total=" << timings.sum() << " secs";

	Utility::WriteToTxt(MuSiKcND[0], "7\\cu.Results.txt");
	Utility::WriteToTxt(MuSiKcND[1], "7\\cu.Rms.txt");
	Utility::WriteToTxt(MuSiKcND[2], "7\\cu.Max.txt");
	Utility::WriteToTxt(MuSiKcND[3], "7\\cu.Timings.txt");

	//wcout << getchar() << endl;
}

int main(int argc, char *argv[]) {

	int experiment = 0;
	if (argc < 2)
	{
		wcout << "Useage: Experiments.exe [ExperimentNumber]" << endl;
		wcout << "ExperimentNumber - Specify an experiment 1-7. Refer to supporting documentation for details of a particular experiment" << endl;
		return -1;
	}
	else
	{
		experiment = std::atoi(argv[1]);
	}

	experiment = 6;

	
	//Experiment2();
	//Experiment3();
	//Experiment5();
	Experiment7();

	//if (experiment == 1)
	//{
	//	Experiment1();
	//}
	//if (experiment == 2)
	//{
	//	Experiment2();
	//}
	//if (experiment == 3)
	//{
	//	Experiment3();
	//}
	//if (experiment == 4)
	//{
	//	Experiment4();
	//}
	//if (experiment == 5)
	//{
	//	Experiment5();
	//}
	//if (experiment == 6)
	//{
	//	Experiment6();
	//}
	//if (experiment == 7)
	//{
	//	Experiment7();
	//}

	wcout << getchar() << endl;
	return 0;
}
