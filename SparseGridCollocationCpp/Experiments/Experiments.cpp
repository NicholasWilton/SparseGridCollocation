// TestHarness.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "SmoothInitialU.h"
#include "SmoothInitialX.h"
#include "Params.h"
#include "BasketOption.h"
#include "Mol.h"
#include "Montecarlo.h"
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
	wcout << "C++ Experiment 1" << endl << endl;
	wcout << "Verify MOL 1D matches MatLab for a European Call Option with 1 asset" << endl << endl;

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
	VectorXd inx1(assets);
	inx1.fill(-p.K);
	p.inx1 = inx1;
	VectorXd inx2(assets);
	inx2.fill(6.0 * p.K);
	p.inx2 = inx2;
	
	p.useCuda = false;
	p.GenerateSmoothInitialUsing = InitialMethod::MethodOfLines;
	p.SmoothInitialPath = "1\\";

	MatrixXd matLabT = Utility::ReadBinary("1", "SmoothInitialT.dat", 1, 1);
	MatrixXd matLabX = Utility::ReadBinary("1", "SmoothInitialX.dat", 32769, 1);
	MatrixXd matLabU = Utility::ReadBinary("1", "SmoothInitialU.dat", 32769, 1);
	
	vector<VectorXd> results = Leicester::SparseGridCollocation::MoL::MethodOfLines(p);

	double precision = 0.00001;
	wcout << "Checking vs MatLab" << endl;

	wcout << "X elements:" << endl;
	if (Utility::checkMatrix(matLabX, results[0], precision, false))
		wcout << "Information: Individual elements match to precision:" << precision << endl;
	else
		wcout << "Warning: Individual elements did not match" << endl;


	wcout << "U elements:" << endl;
	if (Utility::checkMatrix(matLabU, results[1], precision, false))
		wcout << "Success: U elements match to precision:" << precision << endl;
	else
		wcout << "Error: U elements did not match" << endl;


	wcout << "T done:" << endl;
	if (Utility::checkMatrix(matLabT, results[2], 0.001, false))
		wcout << "Information: Tdone matches to precision:" << precision << endl;
	else
		wcout << "Warning: Tdone elements did not match" << endl;

	Utility::WriteToTxt(matLabX, "1\\matlab.X.txt");
	Utility::WriteToTxt(matLabU, "1\\matlab.U.txt");
	Utility::WriteToTxt(matLabT, "1\\matlab.T.txt");
	Utility::WriteToTxt(results[0], "1\\cpp.X.txt");
	Utility::WriteToTxt(results[1], "1\\cpp.U.txt");
	Utility::WriteToTxt(results[2], "1\\cpp.T.txt");


	//wcout << getchar() << endl;
}

void Experiment2()
{
	wcout << "C++ Experiment 2" << endl << endl;
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
	p.useCuda = false;
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
	wcout << "Total=" << timings.sum() << " secs" << endl;

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

	Utility::WriteToTxt(SiKc[0], "2\\cpp.Results.txt");
	Utility::WriteToTxt(SiKc[1], "2\\cpp.Rms.txt");
	Utility::WriteToTxt(SiKc[2], "2\\cpp.Max.txt");
	Utility::WriteToTxt(SiKc[3], "2\\cpp.Timings.txt");

	//wcout << getchar() << endl;
}

void Experiment3()
{
	wcout << "C++ Experiment 3" << endl << endl;
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
	p.useCuda = false;
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
	wcout << "Total=" << timings.sum() << " secs" << endl;

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

	Utility::WriteToTxt(MuSiKc[0], "3\\cpp.Results.txt");
	Utility::WriteToTxt(MuSiKc[1], "3\\cpp.Rms.txt");
	Utility::WriteToTxt(MuSiKc[2], "3\\cpp.Max.txt");
	Utility::WriteToTxt(MuSiKc[3], "3\\cpp.Timings.txt");

	//wcout << getchar() << endl;
}

void Experiment4()
{
	wcout << "C++ Experiment 4" << endl << endl;
	wcout << "Verify MOL N-Dimensions for a Basket Option with 1 asset matches MatLab for a European Call Option with 1 asset" << endl << endl;

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
	VectorXd inx1(assets);
	inx1.fill(-p.K);
	p.inx1 = inx1;
	VectorXd inx2(assets);
	inx2.fill(6.0 * p.K);
	p.inx2 = inx2;
	MatrixXd correlation = MatrixXd::Identity(assets, assets);
	p.useCuda = false;
	p.GenerateSmoothInitialUsing = InitialMethod::MethodOfLines;
	p.SmoothInitialPath = "4\\";

	MatrixXd matLabT = Utility::ReadBinary("4", "SmoothInitialT.dat", 1, 1);
	MatrixXd matLabX = Utility::ReadBinary("4", "SmoothInitialX.dat", 32769, 1);
	MatrixXd matLabU = Utility::ReadBinary("4", "SmoothInitialU.dat", 32769, 1);

	SmoothInitial results = Leicester::SparseGridCollocation::MoL::MethodOfLinesND(p, correlation);

	double precision = 0.00001;
	wcout << "Checking vs MatLab" << endl;

	wcout << "X elements:" << endl;
	if (Utility::checkMatrix(matLabX, results.S, precision, false))
		wcout << "Information: Individual elements match to precision:" << precision << endl;
	else
		wcout << "Warning: Individual elements did not match" << endl;


	wcout << "U elements:" << endl;
	if (Utility::checkMatrix(matLabU, results.U, precision, false))
		wcout << "Success: U elements match to precision:" << precision << endl;
	else
		wcout << "Error: U elements did not match" << endl;


	wcout << "T done:" << endl;
	if (matLabT(0,0) == results.T)
		wcout << "Information: Tdone matches" << endl;
	else
		wcout << "Warning: Tdone elements did not match" << endl;

	Utility::WriteToTxt(matLabX, "4\\matlab.X.txt");
	Utility::WriteToTxt(matLabU, "4\\matlab.U.txt");
	Utility::WriteToTxt(matLabT, "4\\matlab.T.txt");
	Utility::WriteToTxt(results.S, "4\\cpp.X.txt");
	Utility::WriteToTxt(results.U, "4\\cpp.U.txt");
	MatrixXd t(1, 1);
	t << results.T;
	Utility::WriteToTxt(t, "4\\cpp.T.txt");
	//wcout << getchar() << endl;
}

void Experiment5()
{
	wcout << "C++ Experiment 5" << endl << endl;
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
	p.useCuda = false;
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

	Utility::WriteToTxt(MuSiKcND[0], "5\\cpp.Results.txt");
	Utility::WriteToTxt(MuSiKcND[1], "5\\cpp.Rms.txt");
	Utility::WriteToTxt(MuSiKcND[2], "5\\cpp.Max.txt");
	Utility::WriteToTxt(MuSiKcND[3], "5\\cpp.Timings.txt");
	//wcout << getchar() << endl;
}

void Experiment6()
{
	wcout << "C++ Experiment 6" << endl << endl;
	wcout << "Verify MOL N-Dimensions for a Basket Option with 2 assets matches QuantLib Montecarlo" << endl << endl;

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
	VectorXd inx1(assets);
	inx1.fill(-p.K);
	p.inx1 = inx1;
	VectorXd inx2(assets);
	inx2.fill(6.0 * p.K);
	p.inx2 = inx2;
	MatrixXd correlation = MatrixXd::Identity(assets, assets);
	p.useCuda = false;
	p.GenerateSmoothInitialUsing = InitialMethod::MethodOfLines;
	p.SmoothInitialPath = "6\\";

	SmoothInitial quantLib = Leicester::SparseGridCollocation::Montecarlo::BasketOption(p);
	SmoothInitial results = Leicester::SparseGridCollocation::MoL::MethodOfLinesND(p, correlation);

	double precision = 0.00001;
	wcout << "MoL vs Montecarlo" << endl;

	wcout << "X elements:" << endl;
	if (Utility::checkMatrix(quantLib.S, results.S, precision, false))
		wcout << "Information: Individual elements match to precision:" << precision << endl;
	else
		wcout << "Warning: Individual elements did not match" << endl;


	wcout << "U elements:" << endl;
	if (Utility::checkMatrix(quantLib.U, results.U, precision, false))
		wcout << "Success: U elements match to precision:" << precision << endl;
	else
		wcout << "Error: U elements did not match" << endl;


	wcout << "T done:" << endl;
	if (quantLib.T == results.T)
		wcout << "Information: Tdone matches" << endl;
	else
		wcout << "Warning: Tdone elements did not match" << endl;

	Utility::WriteToTxt(quantLib.S, "6\\mc.X.txt");
	Utility::WriteToTxt(quantLib.U, "6\\mc.U.txt");
	MatrixXd qt(1, 1);
	qt << quantLib.T;
	Utility::WriteToTxt(qt, "6\\mc.T.txt");

	Utility::WriteToTxt(results.S, "6\\mol.X.txt");
	Utility::WriteToTxt(results.U, "6\\mol.U.txt");
	MatrixXd mt(1, 1);
	mt << results.T;
	Utility::WriteToTxt(mt, "6\\mol.T.txt");

	//wcout << getchar() << endl;
}

void Experiment7()
{
	wcout << "C++ Experiment 7" << endl << endl;
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
	p.useCuda = false;
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
	wcout << "Total=" << timings.sum() << " secs" << endl;

	Utility::WriteToTxt(MuSiKcND[0], "7\\cpp.Results.txt");
	Utility::WriteToTxt(MuSiKcND[1], "7\\cpp.Rms.txt");
	Utility::WriteToTxt(MuSiKcND[2], "7\\cpp.Max.txt");
	Utility::WriteToTxt(MuSiKcND[3], "7\\cpp.Timings.txt");

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

	//Experiment1();
	//Experiment2();
	//Experiment3();
	//Experiment4();
	//Experiment5();
	//Experiment6();
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
