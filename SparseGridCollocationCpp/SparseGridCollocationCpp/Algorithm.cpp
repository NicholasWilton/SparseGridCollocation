// SparseGridGollocation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialX.h"
#include "SmoothInitialU.h"
#include "Algorithm.h"
#include "EuropeanCallOption.h"
#include "windows.h"
#include ".\..\Common\Utility.h"
#include "Option.h"
#include "BasketOption.h"
#include "VectorUtil.h"
#include "Interpolation.h"
#include "RBF.h"
#include "InterTest.h"
#include <iomanip>
#include "Params.h"
#include "PDE.h"
#include "MoL.h"
//#include "C:\Users\User\Source\Repos\SparseGridCollocation\CudaLib\kernel.h"
#include "SmoothInitial.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using Eigen::Map;
using namespace Eigen;
using namespace std;
using namespace Leicester::Common;

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
//#include <cmath>
//#include <math.h>
#include <thread>


//#include "CppUnitTest.h"
//using namespace Microsoft::VisualStudio::CppUnitTestFramework;

Leicester::SparseGridCollocation::Algorithm::Algorithm()
{
}

Leicester::SparseGridCollocation::Algorithm::~Algorithm()
{
}

map<string, vector<vector<MatrixXd>>> Leicester::SparseGridCollocation::Algorithm::GetInterpolationState()
{
	return InterpolationState;
}

vector<MatrixXd> Leicester::SparseGridCollocation::Algorithm::SIKc(int upper, int lower, Params p)
{
	InterpolationState.clear();

	cout << "SiK-c with levels " << lower << " to " << upper << endl;
	cout << "Parameters:" << endl;
	cout << setprecision(16) << "T=" << p.T << endl;
	cout << setprecision(16) << "Tdone=" << p.Tdone << endl;
	cout << setprecision(16) << "Tend=" << p.Tend << endl;
	cout << setprecision(16) << "dt=" << p.dt << endl;
	cout << setprecision(16) << "K=" << p.K << endl;
	cout << setprecision(16) << "r=" << p.r << endl;
	cout << setprecision(16) << "sigma=" << p.sigma << endl;
	cout << setprecision(16) << "theta=" << p.theta << endl;
	cout << setprecision(16) << "inx1=" << p.inx1 << endl;
	cout << setprecision(16) << "inx2=" << p.inx2 << endl;

	vector<VectorXd> smoothinitial = MoL::MethodOfLines(p);
	SmoothInitialX::x = smoothinitial[0].head(smoothinitial[0].rows());
	SmoothInitialU::u = smoothinitial[1].head(smoothinitial[1].rows());

	p.Tdone = 0.1350;
	VectorXd inx1(1);
	inx1[0] = 0;
	p.inx1 = inx1;
	VectorXd inx2(1);
	inx2[0] = 3.0 * p.K;
	p.inx2 = inx2;

	return SIKc(upper, lower, p, InterpolationState);
}

vector<MatrixXd> Leicester::SparseGridCollocation::Algorithm::SIKc(int upper, int lower, Params p, map<string, vector<vector<MatrixXd>>>& interpolation)
{
	cout << "Starting SiK-c" << endl;

	double E = p.K;// strike price
	double r = p.r; // interest rate
	double sigma = p.sigma;
	double T = p.T; // Maturity
	double inx1 = p.inx1[0];
	double inx2 = p.inx2[0];

	double Tdone = p.Tdone;
	double tsec = T - Tdone; // Initial time boundary for sparse grid
	int d = 2; // dimension
	double coef = 2; // coef stands for the connection constant number

	int ch = 10000;

	VectorXd x = VectorXd::LinSpaced(ch, inx1, inx2);
	VectorXd t = VectorXd::Zero(ch, 1);

	MatrixXd TX(ch, 2);
	TX << t, x;

	int na = 3;
	int nb = 2;

	// Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
	// C stands for shape parater, A stands for scale parameter
	vector<string> level = {};
	if (upper >= 2 & lower <= 2)
	{
		Common::Utility::Logger("level-2");
		string key = "3";
		string _key = "2";
		{
			Interpolation i;
			i.interpolateGeneric(key, coef, tsec, na, d, inx1, inx2, r, sigma, T, E, level, &interpolation);
			interpolation[key] = i.getResult();
		}
		{

			Interpolation i;
			i.interpolateGeneric(_key, coef, tsec, nb, d, inx1, inx2, r, sigma, T, E, level, &interpolation);
			interpolation[_key] = i.getResult();
		}
	}
	//Multi level interpolation requires successive reuse of results from prior levels
	for (int count = 3; count <= 10; count++)
	{
		stringstream ss;
		ss << "level-" << count;
		Common::Utility::Logger(ss.str());
		stringstream ss1;
		ss1 << count + 1;
		string key = ss1.str();
		stringstream ss2;
		ss2 << "_" << count;
		string _key = ss2.str();
		{
			Interpolation i;
			i.interpolateGeneric(key, coef, tsec, na + (count - 2), d, inx1, inx2, r, sigma, T, E, level, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			i.interpolateGeneric(_key, coef, tsec, nb + (count - 2), d, inx1, inx2, r, sigma, T, E, level, &interpolation);
			interpolation[_key] = i.getResult();
		}

	}

	Common::Utility::Logger("inter_test");

	InterTest interTest;
	vector<thread> threads;
	vector<InterTest> interTests;

	if (upper >= 2 & lower <= 2)
	{
		InterTest interTest2;
		InterTest interTest3;

		vector<vector<MatrixXd>> test2 = interpolation["2"];
		threads.push_back(std::thread(&InterTest::parallel, interTest2, "2", TX, test2[0], test2[1], test2[2], test2[3]));
		vector<vector<MatrixXd>> test3 = interpolation["3"];
		threads.push_back(std::thread(&InterTest::parallel, interTest3, "3", TX, test3[0], test3[1], test3[2], test3[3]));
		interTests.push_back(interTest2);
		interTests.push_back(interTest3);
	}
	for (int count = 3; count <= 10; count++)
	{
		if (upper >= count & lower <= count)
		{
			InterTest interTest_;
			InterTest interTest;
			stringstream ss1;
			ss1 << "_" << count;
			vector<vector<MatrixXd>> test_ = interpolation[ss1.str()];
			threads.push_back(std::thread(&InterTest::parallel, interTest_, ss1.str(), TX, test_[0], test_[1], test_[2], test_[3]));
			stringstream ss2;
			ss2 << count + 1;
			vector<vector<MatrixXd>> test = interpolation[ss2.str()];
			threads.push_back(std::thread(&InterTest::parallel, interTest, ss2.str(), TX, test[0], test[1], test[2], test[3]));
			interTests.push_back(interTest_);
			interTests.push_back(interTest);
		}
	}

	for (int i = 0; i < threads.size(); i++)
		threads.at(i).join();

	Common::Utility::Logger("inter_test complete");
	MatrixXd SiK(ch, upper - 3);

	if (upper >= 2 & lower <= 2)
	{
		MatrixXd V3 = interTests.at(1).GetResult("3");
		MatrixXd V_2 = interTests.at(0).GetResult("2");
		SiK.col(0) = V3 - V_2;
	}
	int index = 2;
	for (int count = 1; count < SiK.cols(); count++, index += 2)
	{
		stringstream ss1;
		ss1 << count + 3;
		MatrixXd V = interTests.at(index + 1).GetResult(ss1.str());
		stringstream ss2;
		ss2 << "_" << count + 2;
		MatrixXd V_ = interTests.at(index).GetResult(ss2.str());
		SiK.col(count) = V - V_;
	}

	Option *option = new EuropeanCallOption(E, T);
	VectorXd AP = option->Price(TX, r, sigma);
	delete option;

	VectorXd RMS = VectorXd::Ones(9, 1);
	VectorXd Max = VectorXd::Ones(9, 1);

	Common::Utility::Logger("Error calculations");
	for (int i = 0; i < SiK.cols(); i++)
	{
		VectorXd v = SiK.col(i).array() - AP.array();
		RMS[i] = RootMeanSquare(v);
		VectorXd m = abs(SiK.col(i).array() - AP.array());
		Max[i] = m.maxCoeff();
	}

	InterpolationState = interpolation;
	vector<MatrixXd> result = { SiK, RMS, Max };
	return result;
}

vector<MatrixXd> Leicester::SparseGridCollocation::Algorithm::MuSIKc(int upper, int lower, Params p)
{
	InterpolationState.clear();

	cout << "MuSiK-c with levels " << lower << " to " << upper << endl;
	cout << "Parameters:" << endl;
	cout << setprecision(16) << "T=" << p.T << endl;
	cout << setprecision(16) << "Tdone=" << p.Tdone << endl;
	cout << setprecision(16) << "Tend=" << p.Tend << endl;
	cout << setprecision(16) << "dt=" << p.dt << endl;
	cout << setprecision(16) << "K=" << p.K << endl;
	cout << setprecision(16) << "r=" << p.r << endl;
	cout << setprecision(16) << "sigma=" << p.sigma << endl;
	cout << setprecision(16) << "theta=" << p.theta << endl;
	cout << setprecision(16) << "inx1=" << p.inx1 << endl;
	cout << setprecision(16) << "inx2=" << p.inx2 << endl;

	int old1 = p.inx1[0];
	int old2 = p.inx2[0];
	p.inx1[0] = -p.K;
	p.inx2[0] = 6.0 * p.K;

	vector<VectorXd> smoothinitial = MoL::MethodOfLines(p);
	SmoothInitialX::x = smoothinitial[0].head(smoothinitial[0].rows());
	SmoothInitialU::u = smoothinitial[1].head(smoothinitial[1].rows());
	p.Tdone = smoothinitial[2][0];
	//p.Tdone = 0.1350;
	cout << setprecision(16) << "Tdone=" << p.Tdone << endl;
	p.inx1[0] = old1;
	p.inx2[0] = old2;

	return MuSIKc(upper, lower, p, InterpolationState);
}

vector<MatrixXd> Leicester::SparseGridCollocation::Algorithm::MuSIKc(int upper, int lower, Params p, map<string, vector<vector<MatrixXd>>>& interpolation)
{
	cout << "Starting MuSiK-c" << endl;

	double E = p.K;// strike price
	double r = p.r; // interest rate
	double sigma = p.sigma;
	double T = p.T; // Maturity
	double inx1 = p.inx1[0];
	double inx2 = p.inx2[0];
	double Tdone = p.Tdone;
	double tsec = T - Tdone; // Initial time boundary for sparse grid
	int d = 2; // dimension
	double coef = 2; // coef stands for the connection constant number

	int ch = 10000;

	VectorXd x = VectorXd::LinSpaced(ch, inx1, inx2);
	VectorXd t = VectorXd::Zero(ch, 1);

	MatrixXd TX(ch, 2);
	TX << t, x;

	int na = 3;
	int nb = 2;

	// Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
	// C stands for shape parater, A stands for scale parameter
	vector<string> level = {};
	if (upper >= 2 & lower <= 2)
	{
		Common::Utility::Logger("level-2");
		string key = "3";
		string _key = "2";
		{
			Interpolation i;
			i.interpolateGeneric(key, coef, tsec, na, d, inx1, inx2, r, sigma, T, E, level, &interpolation);
			interpolation[key] = i.getResult();
		}
		{

			Interpolation i;
			i.interpolateGeneric(_key, coef, tsec, nb, d, inx1, inx2, r, sigma, T, E, level, &interpolation);
			interpolation[_key] = i.getResult();
		}
		level.push_back(_key);
		level.push_back(key);
	}
	//Multi level interpolation requires successive reuse of results from prior levels
	for (int count = 3; count <= 10; count++)
	{
		if (upper >= count & lower <= count)
		{
			stringstream ss;
			ss << "level-" << count;
			Common::Utility::Logger(ss.str());
			stringstream ss1;
			ss1 << count + 1;
			string key = ss1.str();
			stringstream ss2;
			ss2 << "_" << count;
			string _key = ss2.str();
			{
				Interpolation i;
				i.interpolateGeneric(key, coef, tsec, na + (count - 2), d, inx1, inx2, r, sigma, T, E, level, &interpolation);
				interpolation[key] = i.getResult();
			}
			{
				Interpolation i;
				i.interpolateGeneric(_key, coef, tsec, nb + (count - 2), d, inx1, inx2, r, sigma, T, E, level, &interpolation);
				interpolation[_key] = i.getResult();
			}
			level.push_back(_key);
			level.push_back(key);
		}
	}

	//for (auto i : interpolation)
	//{
	//	stringstream ss;
	//	ss << "MuSiKc_" << i.first;
	//	int countj = 0;
	//	for (auto j : i.second)
	//	{
	//		stringstream ssj;
	//		ssj << ss.str() << "." << countj;
	//		int countk = 0;
	//		for (auto k : j)
	//		{
	//			stringstream ssk;
	//			ssk << ssj.str() << "." << countk << ".txt";
	//			Common::Utility::saveArray(k, ssk.str());
	//			countk++;
	//		}
	//		countj++;
	//	}
	//}

	Common::Utility::Logger("inter_test");

	InterTest interTest;
	vector<thread> threads;
	vector<InterTest> interTests;

	if (upper >= 2 & lower <= 2)
	{
		InterTest interTest2;
		InterTest interTest3;

		vector<vector<MatrixXd>> test2 = interpolation["2"];
		threads.push_back(std::thread(&InterTest::parallel, interTest2, "2", TX, test2[0], test2[1], test2[2], test2[3]));
		vector<vector<MatrixXd>> test3 = interpolation["3"];
		threads.push_back(std::thread(&InterTest::parallel, interTest3, "3", TX, test3[0], test3[1], test3[2], test3[3]));
		interTests.push_back(interTest2);
		interTests.push_back(interTest3);
	}
	for (int count = 3; count <= 10; count++)
	{
		if (upper >= count & lower <= count)
		{
			InterTest interTest_;
			InterTest interTest;
			stringstream ss1;
			ss1 << "_" << count;
			vector<vector<MatrixXd>> test_ = interpolation[ss1.str()];
			threads.push_back(std::thread(&InterTest::parallel, interTest_, ss1.str(), TX, test_[0], test_[1], test_[2], test_[3]));
			stringstream ss2;
			ss2 << count + 1;
			vector<vector<MatrixXd>> test = interpolation[ss2.str()];
			threads.push_back(std::thread(&InterTest::parallel, interTest, ss2.str(), TX, test[0], test[1], test[2], test[3]));
			interTests.push_back(interTest_);
			interTests.push_back(interTest);
			//cout << "key:" << ss1.str() << endl;
			//cout << "key:" << ss2.str() << endl;
		}
	}

	for (int i = 0; i < threads.size(); i++)
		threads.at(i).join();

	//int counti = 0;
	//for (auto i : interTests)
	//{
	//	stringstream ss;
	//	ss << "MuSiKc_InterTest_" << counti;
	//	int countj = 0;
	//	for (auto j : i.GetResults())
	//	{
	//		stringstream ssj;
	//		ssj << ss.str() << "." << j.first << ".txt";

	//		Common::Utility::saveArray(j.second, ssj.str());

	//		countj++;
	//	}
	//	counti++;
	//}

	Common::Utility::Logger("inter_test complete");
	vector<VectorXd> Us = { VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000),VectorXd::Zero(10000) };

	if (upper >= 2 & lower <= 2)
	{
		MatrixXd V3 = interTests.at(1).GetResult("3");
		MatrixXd V_2 = interTests.at(0).GetResult("2");
		Us[0] = V3 - V_2;
	}
	int index = 2;
	for (int count = 3; count <= 10; count++, index += 2)
	{
		if (upper >= count & lower <= count)
		{
			stringstream ss1;
			ss1 << count + 1;
			MatrixXd V = interTests.at(index + 1).GetResult(ss1.str());
			//cout << V(1, 0) << endl;
			stringstream ss2;
			ss2 << "_" << count;
			MatrixXd V_ = interTests.at(index).GetResult(ss2.str());
			//cout << V_(1, 0) << endl;
			Us[count - 2] = V - V_;
			//cout << Us[count - 2](1, 0) << endl;
			//wcout << Common::printMatrix(Us[count - 2]) << endl;
		}
	}

	//int countu = 0;
	//for (auto u : Us)
	//{
	//	stringstream ss;
	//	ss << "musikc_U." << countu << ".txt";
	//	Common::saveArray(u, ss.str());
	//	countu++;
	//}

	EuropeanCallOption option(E, T);
	VectorXd AP = option.Price(TX, r, sigma);
	Common::Utility::Logger("MuSIK addition");
	int m = Us[0].rows();
	MatrixXd MuSIK = MatrixXd::Zero(m, 9);
	VectorXd sum = VectorXd::Zero(Us[0].rows());
	for (int count = 2; count <= 10; count++)
	{
		sum = sum + Us[count - 2];
		if (upper >= count & lower <= count)
		{
			MuSIK.col(count - 2) = sum;
		}
	}

	VectorXd RMS = VectorXd::Ones(9, 1);
	VectorXd Max = VectorXd::Ones(9, 1);

	Common::Utility::Logger("Error calculations");
	for (int i = 0; i < MuSIK.cols(); i++)
	{
		VectorXd v = MuSIK.col(i).array() - AP.array();
		//stringstream ss;
		//ss << "musikc_v." << i << ".txt";
		//Common::saveArray(v, ss.str());
		RMS[i] = RootMeanSquare(v);
		VectorXd m = abs(MuSIK.col(i).array() - AP.array());
		Max[i] = m.maxCoeff();
	}

	InterpolationState = interpolation;
	vector<MatrixXd> result = { MuSIK, RMS, Max };
	return result;
}

vector<MatrixXd> Leicester::SparseGridCollocation::Algorithm::MuSIKcND(int upper, int lower, BasketOption option, Params p)
{
	InterpolationState.clear();

	cout << "N-Dimensional MuSiK-c with levels " << lower << " to " << upper << endl;
	cout << "Parameters:" << endl;
	cout << setprecision(16) << "T=" << p.T << endl;
	cout << setprecision(16) << "Tend=" << p.Tend << endl;
	cout << setprecision(16) << "dt=" << p.dt << endl;
	cout << setprecision(16) << "K=" << p.K << endl;
	cout << setprecision(16) << "r=" << p.r << endl;
	cout << setprecision(16) << "sigma=" << p.sigma << endl;
	cout << setprecision(16) << "theta=" << p.theta << endl;
	cout << setprecision(16) << "inx1=" << p.inx1 << endl;
	cout << setprecision(16) << "inx2=" << p.inx2 << endl;

	VectorXd old1 = p.inx1;
	VectorXd old2 = p.inx2;
	p.inx1.fill(-p.K);
	p.inx2.fill(6.0 * p.K);

	SmoothInitial smoothinitial = MoL::MethodOfLinesND(p, option.correlation);
	SmoothInitialX::x = smoothinitial.S;
	SmoothInitialU::u = smoothinitial.U;

	p.Tdone = smoothinitial.T;
	//p.Tdone = 0.1350;
	cout << setprecision(16) << "Tdone=" << p.Tdone << endl;
	p.inx1 = old1;
	p.inx2 = old2;

	return MuSIKcND(upper, lower, option, p, InterpolationState);
}

vector<MatrixXd> Leicester::SparseGridCollocation::Algorithm::MuSIKcND(int upper, int lower, BasketOption option, Params p, map<string, vector<vector<MatrixXd>>>& interpolation)
{
	Interpolation::callCount = 0;
	PDE::callCount = 0;
	cout << "Starting MuSiK-c" << endl;
	wcout << "Interpolation Call count=" << Interpolation::callCount << endl;
	wcout << "PDE Call count=" << Interpolation::callCount << endl;

	int dimensions = option.Underlying + 1; // 1 per asset + time
	double E = p.K;// strike price
	double r = p.r; // interest rate
	double sigma = p.sigma;
	double T = p.T; // Maturity
	MatrixXd inx1(1, option.Underlying);
	MatrixXd inx2(1, option.Underlying);
	for (int col = 0; col < inx1.cols(); col++)
	{
		inx1(0, col) = p.inx1[col];
		inx2(0, col) = p.inx2[col];
	}

	double Tdone = p.Tdone;
	double tsec = T - Tdone; // Initial time boundary for sparse grid
	int d = 2; // dimension
	double coef = 2; // coef stands for the connection constant number

	int ch = 10000; //in 4d heat this is set to 22

	MatrixXd TestGrid(ch, dimensions);
	TestGrid.col(0) = VectorXd::Zero(ch);
	for (int d = 1; d < dimensions; d++)
		TestGrid.col(d) = VectorXd::LinSpaced(ch, inx1(0, d - 1), inx2(0, d - 1));

	int lvl = dimensions;

	// Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
	// C stands for shape parater, A stands for scale parameter
	vector<string> level = {};
	if (upper >= lvl & lower <= lvl)
	{
		vector<string> newKeys = {};
		stringstream ss;
		ss << "level-" << lvl;
		Common::Utility::Logger(ss.str());
		ss.str(string());
		ss << 1 + lvl;

		int n = lvl + option.Underlying;
		string key = ss.str();
		{
			Interpolation i;
			i.interpolateGenericND(key, coef, tsec, n, dimensions, inx1, inx2, r, sigma, T, E, level, &interpolation, p.useCuda);
			interpolation[key] = i.getResult();
		}
		newKeys.push_back(key);

		for (int dimension = 1; dimension <= option.Underlying; dimension++)
		{
			n = lvl + option.Underlying - dimension;
			stringstream ss;
			ss << 1 + lvl << "_" << n;
			string key = ss.str();
			{
				Interpolation i;
				i.interpolateGenericND(key, coef, tsec, n, dimensions, inx1, inx2, r, sigma, T, E, level, &interpolation, p.useCuda);
				interpolation[key] = i.getResult();
			}
			newKeys.push_back(key);
		}
		for (auto key : newKeys)
			level.push_back(key);
	}
	lvl++;

	//Multi level interpolation requires successive reuse of results from prior levels
	int sequence = 1;
	for (lvl; lvl <= 10 + dimensions; lvl++)
	{
		if (lvl <= upper & lvl >= lower)
		{
			vector<string> newKeys = {};
			stringstream ss;
			ss << "level-" << lvl;
			Common::Utility::Logger(ss.str());

			stringstream ss1;
			ss1 << 1 + lvl;
			int n = lvl + option.Underlying;
			string key = ss1.str();
			{
				Interpolation i;
				i.interpolateGenericND(key, coef, tsec, n, dimensions, inx1, inx2, r, sigma, T, E, level, &interpolation, p.useCuda);
				interpolation[key] = i.getResult();
			}
			newKeys.push_back(key);
			for (int dimension = 1; dimension <= option.Underlying; dimension++)
			{
				n = lvl + option.Underlying - dimension;
				stringstream ss2;
				ss2 << 1 + lvl << "_" << n;
				string key = ss2.str();
				{
					Interpolation i;
					i.interpolateGenericND(key, coef, tsec, n, dimensions, inx1, inx2, r, sigma, T, E, level, &interpolation, p.useCuda);
					interpolation[key] = i.getResult();
				}
				newKeys.push_back(key);
			}
			for (auto key : newKeys)
				level.push_back(key);

			sequence++;
		}
	}
	wcout << "Interpolation Call count=" << Interpolation::callCount << endl;
	wcout << "PDE Call count=" << Leicester::SparseGridCollocation::PDE::callCount << endl;
	//for (auto i : interpolation)
	//{
	//	stringstream ss;
	//	ss << "MuSiKcND_" << i.first;
	//	int countj = 0;
	//	for (auto j : i.second)
	//	{
	//		stringstream ssj;
	//		ssj << ss.str() << "." << countj;
	//		int countk = 0;
	//		for (auto k : j)
	//		{
	//			stringstream ssk;
	//			ssk << ssj.str() << "." << countk << ".txt";
	//			Common::Utility::saveArray(k, ssk.str());
	//			countk++;
	//		}
	//		countj++;
	//	}
	//}
	Common::Utility::Logger("inter_test");

	InterTest interTest;
	vector<thread> threads;
	vector<InterTest> interTests;

	for (int count = dimensions; count <= 10 + dimensions; count++)
	{
		if (upper >= count & lower <= count)
		{
			int lvl = count;
			int n = lvl + option.Underlying;
			InterTest interTest;
			stringstream ss1;
			ss1 << n;
			vector<vector<MatrixXd>> test_ = interpolation[ss1.str()];
			threads.push_back(std::thread(&InterTest::parallelND, interTest, ss1.str(), TestGrid, test_[0], test_[1], test_[2], test_[3]));
			//interTest.parallelND(ss1.str(), TestGrid, test_[0], test_[1], test_[2], test_[3]);
			//cout << "key:" << ss1.str() << endl;
			interTests.push_back(interTest);

			for (int dimension = 1; dimension <= option.Underlying; dimension++)
			{
				InterTest interTest_;
				stringstream ss2;
				ss2 << n << "_" << n - dimension;
				vector<vector<MatrixXd>> test = interpolation[ss2.str()];
				threads.push_back(std::thread(&InterTest::parallelND, interTest_, ss2.str(), TestGrid, test[0], test[1], test[2], test[3]));

				//interTest_.parallelND(ss2.str(), TestGrid, test[0], test[1], test[2], test[3]);
				interTests.push_back(interTest_);
				//cout << "key:" << ss2.str() << endl;
				dimension++;
			}


		}
	}

	for (int i = 0; i < threads.size(); i++)
		threads.at(i).join();

	//int counti = 0;
	//for (auto i : interTests)
	//{
	//	stringstream ss;
	//	ss << "MuSiKcND_InterTest_" << counti;
	//	int countj = 0;
	//	for (auto j : i.GetResults())
	//	{
	//		stringstream ssj;
	//		ssj << ss.str() << "." << j.first << ".txt";

	//		Common::Utility::saveArray(j.second, ssj.str());

	//		countj++;
	//	}
	//	counti++;
	//}

	Common::Utility::Logger("inter_test complete");
	vector<VectorXd> Us = { VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000),VectorXd::Zero(10000) };

	//TODO: calculate upper limit from portfolio size and level choice

	int n = dimensions;
	int udx = 0;
	//cout << "intertests size=" << interTests.size() << endl;
	//cout << "upper=" << upper << " lower=" << lower << endl;

	for (int count = 0; count < interTests.size(); count += dimensions, udx++)
		//for (int count = dimensions; count <= 10 + dimensions; count ++)
	{
		//cout << "count\idx=" << count << endl;
		//cout << "udx=" << udx << endl;
		int idx = count;
		if (upper >= udx + dimensions & lower <= udx + dimensions)
		{

			n++;
			vector<MatrixXd> elements;
			stringstream ss;
			ss << n;

			MatrixXd U = interTests.at(idx).GetResult(ss.str());
			//wcout << U(1,0) << endl;
			int coeff = Leicester::Common::Utility::BinomialCoefficient(option.Underlying, 0);
			U = U * coeff;
			//wcout << U(1, 0) << endl;
			int index = 1;
			for (int dimension = 1; dimension <= option.Underlying; dimension++)
			{
				ss.str(string());
				ss << n << "_" << n - dimension;
				MatrixXd V = interTests.at(idx + index).GetResult(ss.str());
				//wcout << V(1, 0) << endl;
				coeff = (index % 2 == 0) ? Leicester::Common::Utility::BinomialCoefficient(option.Underlying, dimension) : -Leicester::Common::Utility::BinomialCoefficient(option.Underlying, dimension);
				U = U + (coeff * V);
				//wcout << U(1, 0) << endl;
				index++;

			}
			Us[udx] = U;
		}
	}

	//int countu = 0;
	//for (auto u : Us)
	//{
	//	stringstream ss;
	//	ss << "musikcND_U." << countu << ".txt";
	//	Common::saveArray(u, ss.str());
	//	countu++;
	//}

	//TODO: basket option analytic price.
	VectorXd AP = option.Price(TestGrid, r, sigma);

	Common::Utility::Logger("MuSIK addition");
	int m = Us[0].rows();
	MatrixXd MuSIK = MatrixXd::Zero(m, 9);
	VectorXd sum = VectorXd::Zero(Us[0].rows());
	for (int count = 2; count <= 10; count++)
	{
		sum = sum + Us[count - 2];
		if (upper >= count & lower <= count)
		{
			MuSIK.col(count - 2) = sum;
		}
	}

	VectorXd RMS = VectorXd::Ones(9, 1);
	VectorXd Max = VectorXd::Ones(9, 1);

	Common::Utility::Logger("Error calculations");
	for (int i = 0; i < MuSIK.cols(); i++)
	{
		VectorXd v = MuSIK.col(i).array() - AP.col(0).array();
		//stringstream ss;
		//ss << "musikcND_v." << i << ".txt";
		//Common::saveArray(v, ss.str());

		RMS[i] = RootMeanSquare(v);
		VectorXd m = abs(MuSIK.col(i).array() - AP.col(0).array());
		Max[i] = m.maxCoeff();
	}

	InterpolationState = interpolation;
	vector<MatrixXd> result = { MuSIK, RMS, Max };
	return result;
}



double Leicester::SparseGridCollocation::Algorithm::RootMeanSquare(VectorXd v)
{
	double rms = sqrt((v.array() * v.array()).sum() / v.size());
	return rms;
}



