// SparseGridGollocation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialX.h"
#include "SmoothInitialU.h"
#include "SparseGridCollocation.h"
#include "EuropeanCallOption.h"
#include "windows.h"
#include "Common.h"
#include "Option.h"
#include "BasketOption.h"
#include "VectorUtil.h"
#include "Interpolation.h"
#include "RBF.h"
#include "InterTest.h"
#include <iomanip>
#include "Params.h"
#include "MoL.h"
#include "C:\Users\User\Source\Repos\SparseGridCollocation\CudaLib\kernel.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using Eigen::Map;
using namespace Eigen;
using namespace std;

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <math.h>
#include <thread>


//#include "CppUnitTest.h"
//using namespace Microsoft::VisualStudio::CppUnitTestFramework;

SparseGridCollocation::SparseGridCollocation()
{
}

map<string, vector<vector<MatrixXd>>> SparseGridCollocation::GetInterpolationState()
{
	return InterpolationState;
}

vector<MatrixXd> SparseGridCollocation::SIKc(int upper, int lower, Params p)
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
	p.inx1 = 0;
	p.inx2 = 3.0 * p.K;

	return SIKc(upper, lower, p, InterpolationState);
}

vector<MatrixXd> SparseGridCollocation::SIKc(int upper, int lower, Params p, map<string, vector<vector<MatrixXd>>>& interpolation)
{
	cout << "Starting SiK-c" << endl;

	double E = p.K;// strike price
	double r = p.r; // interest rate
	double sigma = p.sigma;
	double T = p.T; // Maturity
	double inx1 = p.inx1;
	double inx2 = p.inx2;
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
		Common::Logger("level-2");
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
		Common::Logger(ss.str());
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

	Common::Logger("inter_test");

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

	Common::Logger("inter_test complete");
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

	Common::Logger("Error calculations");
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

vector<MatrixXd> SparseGridCollocation::MuSIKc(int upper, int lower, Params p)
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

	vector<VectorXd> smoothinitial = MoL::MethodOfLines(p);
	SmoothInitialX::x = smoothinitial[0].head(smoothinitial[0].rows());
	SmoothInitialU::u = smoothinitial[1].head(smoothinitial[1].rows());

	p.Tdone = 0.1350;
	p.inx1 = 0;
	p.inx2 = 3.0 * p.K;

	return MuSIKc(upper, lower, p, InterpolationState);
}

vector<MatrixXd> SparseGridCollocation::MuSIKc(int upper, int lower, Params p, map<string, vector<vector<MatrixXd>>>& interpolation)
{
	cout << "Starting MuSiK-c" << endl;

	double E = p.K;// strike price
	double r = p.r; // interest rate
	double sigma = p.sigma;
	double T = p.T; // Maturity
	double inx1 = p.inx1;
	double inx2 = p.inx2;
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
		Common::Logger("level-2");
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
		stringstream ss;
		ss << "level-" << count;
		Common::Logger(ss.str());
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

	Common::Logger("inter_test");

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

	Common::Logger("inter_test complete");
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
		stringstream ss1;
		ss1 << count + 1;
		MatrixXd V = interTests.at(index + 1).GetResult(ss1.str());
		stringstream ss2;
		ss2 << "_" << count;
		MatrixXd V_ = interTests.at(index).GetResult(ss2.str());
		Us[count - 2] = V - V_;
	}

	EuropeanCallOption option(E, T);
	VectorXd AP = option.Price(TX, r, sigma);
	Common::Logger("MuSIK addition");
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

	Common::Logger("Error calculations");
	for (int i = 0; i < MuSIK.cols(); i++)
	{
		VectorXd v = MuSIK.col(i).array() - AP.array();
		RMS[i] = RootMeanSquare(v);
		VectorXd m = abs(MuSIK.col(i).array() - AP.array());
		Max[i] = m.maxCoeff();
	}

	InterpolationState = interpolation;
	vector<MatrixXd> result = { MuSIK, RMS, Max };
	return result;
}

vector<MatrixXd> SparseGridCollocation::MuSIKcND(int upper, int lower, BasketOption option, Params p)
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

	vector<VectorXd> smoothinitial = MoL::MethodOfLines(p);
	SmoothInitialX::x = smoothinitial[0].head(smoothinitial[0].rows());
	SmoothInitialU::u = smoothinitial[1].head(smoothinitial[1].rows());

	p.Tdone = 0.1350;
	p.inx1 = 0;
	p.inx2 = 3.0 * p.K;

	return MuSIKcND(upper, lower, option, p, InterpolationState);
}

vector<MatrixXd> SparseGridCollocation::MuSIKcND(int upper, int lower, BasketOption option, Params p, map<string, vector<vector<MatrixXd>>>& interpolation)
{
	cout << "Starting MuSiK-c" << endl;

	int dimensions = option.Underlying + 1; // 1 per asset + time
	double E = p.K;// strike price
	double r = p.r; // interest rate
	double sigma = p.sigma;
	double T = p.T; // Maturity
	MatrixXd inx1(1, dimensions);
	inx1.fill(p.inx1);
	
	MatrixXd inx2(1, dimensions);
	inx2.fill(p.inx2);

	double Tdone = p.Tdone;
	double tsec = T - Tdone; // Initial time boundary for sparse grid
	int d = 2; // dimension
	double coef = 2; // coef stands for the connection constant number

	int ch = 10000; //in 4d heat this is set to 22

	MatrixXd TestGrid(ch, dimensions);
	for (int d = 0; d < dimensions; d++)
		TestGrid.col(d) = VectorXd::LinSpaced(ch, inx1(0, d), inx2(0, d));

	int lvl = dimensions;

	// Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
	// C stands for shape parater, A stands for scale parameter
	vector<string> level = {};
	if (upper >= lvl & lower <= lvl)
	{
		vector<string> newKeys = {};
		Common::Logger("level-4");
		stringstream ss;
		ss << 1 + lvl;
		int n = lvl + option.Underlying;
		string key = ss.str();
		{
			Interpolation i;
			i.interpolateGenericND(key, coef, tsec, n, d, inx1, inx2, r, sigma, T, E, level, &interpolation);
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
				i.interpolateGenericND(key, coef, tsec, n, d, inx1, inx2, r, sigma, T, E, level, &interpolation);
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
	for (lvl; lvl <= 10; lvl++)
	{
		vector<string> newKeys = {};
		stringstream ss;
		ss << "level-" << lvl;
		Common::Logger(ss.str());

		stringstream ss1;
		ss1 << 1 + lvl;
		int n = lvl + option.Underlying;
		string key = ss1.str();
		{
			Interpolation i;
			i.interpolateGenericND(key, coef, tsec, n, d, inx1, inx2, r, sigma, T, E, level, &interpolation);
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
				i.interpolateGenericND(key, coef, tsec, n, d, inx1, inx2, r, sigma, T, E, level, &interpolation);
				interpolation[key] = i.getResult();
			}
			newKeys.push_back(key);
		}
		for (auto key : newKeys)
			level.push_back(key);

		sequence++;
	}
	
	Common::Logger("inter_test");

	InterTest interTest;
	vector<thread> threads;
	vector<InterTest> interTests;

	for (int count = 0; count <= 10; count++)
	{
		if (upper >= count & lower <= count)
		{
			int lvl = 4 + count;
			int n = lvl + option.Underlying;
			InterTest interTest;
			stringstream ss1;
			ss1 << n;
			vector<vector<MatrixXd>> test_ = interpolation[ss1.str()];
			threads.push_back(std::thread(&InterTest::parallelND, interTest, ss1.str(), TestGrid, test_[0], test_[1], test_[2], test_[3]));
			interTests.push_back(interTest);
			
			for (int dimension = 1 ; dimension <= option.Underlying; dimension++)
			{
				InterTest interTest_;
				stringstream ss2;
				ss2 << n << "_" << n - dimension;
				vector<vector<MatrixXd>> test = interpolation[ss2.str()];
				threads.push_back(std::thread(&InterTest::parallelND, interTest_, ss2.str(), TestGrid, test[0], test[1], test[2], test[3]));
				interTests.push_back(interTest_);
				dimension++;
			}
			
			
		}
	}

	for (int i = 0; i < threads.size(); i++)
		threads.at(i).join();

	Common::Logger("inter_test complete");
	vector<VectorXd> Us = { VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000) ,VectorXd::Zero(10000),VectorXd::Zero(10000) };

	//TODO: calculate upper limit from portfolio size and level choice
	
	for (int count = 0; count <= 10; count+= option.Underlying + 1)
	{
		int lvl = 4 + count;
		int n = lvl + option.Underlying;
		vector<MatrixXd> elements;
		stringstream ss;
		ss << n;
		MatrixXd U = interTests.at(count).GetResult(ss.str());
		int coeff = Common::BinomialCoefficient(option.Underlying + 1, 0);
		U = U * coeff;
		int index = 1;
		for (int dimension =1; dimension <= option.Underlying; dimension ++)
		{
			stringstream ss1;
			ss1 << count + 1;
			MatrixXd V = interTests.at(count + index).GetResult(ss1.str());
			coeff = (index % 2 == 0) ? Common::BinomialCoefficient(option.Underlying + 1, count) : -Common::BinomialCoefficient(option.Underlying + 1, count);
			U = U + (coeff * V);
			index++;
		}
		Us[count] = U;
	}
	
	//TODO: basket option analytic price.
	MatrixXd AP = option.Price(TestGrid.col(0), r, sigma);
	
	Common::Logger("MuSIK addition");
	int m = Us[0].rows();
	MatrixXd MuSIK = MatrixXd::Zero(m,9);
	VectorXd sum = VectorXd::Zero(Us[0].rows());
	for (int count = 2; count <= 10; count++)
	{
		sum = sum + Us[count - 2];
		if (upper >= count & lower <= count)
		{
			MuSIK.col(count - 2) = sum;
		}
	}

	VectorXd RMS = VectorXd::Ones(9,1);
	VectorXd Max = VectorXd::Ones(9, 1);

	Common::Logger("Error calculations");
	//for (int i = 0; i < MuSIK.cols(); i++)
	//{
	//	VectorXd v = MuSIK.col(i).array() - AP.array();
	//	RMS[i] = RootMeanSquare(v);
	//	VectorXd m = abs(MuSIK.col(i).array() - AP.array());
	//	Max[i] = m.maxCoeff();
	//}

	InterpolationState = interpolation;
	vector<MatrixXd> result = { MuSIK, RMS, Max };
	return result;
}



double SparseGridCollocation::RootMeanSquare(VectorXd v)
{
	double rms = sqrt((v.array() * v.array()).sum() / v.size() );
	return rms;
}



