// SparseGridGollocation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialX.h"
#include "SparseGridCollocation.h"
#include "windows.h"
#include "Common.h"
#include "Interpolation.h"
#include "RBF.h"
#include "InterTest.h"


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
	return vInterpolation;
}

map<int,MatrixXd> SparseGridCollocation::GetU()
{
	return uMatrix;
}

MatrixXd SparseGridCollocation::ECP(const MatrixXd &X, double r, double sigma, double T, double E)
{
	VectorXd t= X.col(0);
	VectorXd S = X.col(1);
	VectorXd M = T - t.array();
	int N = X.rows();
	VectorXd P = VectorXd::Ones(N, 1);
	VectorXd d1 = VectorXd::Ones(N, 1);
	VectorXd d2 = VectorXd::Ones(N, 1);

	MatrixXd I0 = (M.array() == 0).select(M,0);
	MatrixXd I1 = (M.array() != 0).select(M,0);

	for (int i = 0; i < N; i++)
	{
		if (S(i) - E > 0)
			P(i) = S(i) - E;
		else
			P(i) = 0;

		d1(i) = (log(S(i) / E) + (r + sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		d2(i) = (log(S(i) / E) + (r - sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		P(i) = S(i) * normCDF(d1(i)) - E * exp(-r * M(i)) * normCDF(d2(i));
	}
	return P;
}

double SparseGridCollocation::normCDF(double value)
{
	return 0.5 * erfc(-value * (1/sqrt(2)));
}

vector<MatrixXd> SparseGridCollocation::MuSIKGeneric(int upper, int lower)
{
	return MuSIKGeneric(upper, lower, vInterpolation);
}
vector<MatrixXd> SparseGridCollocation::MuSIKGeneric(int upper, int lower, map<string, vector<vector<MatrixXd>>>& interpolation)
{
	double E = 100;// strike price

	double r = 0.03; // interest rate
	double sigma = 0.15;
	double T = 1; // Maturity
	double inx1 = 0; // stock price S belongs to[inx1 inx2]
	double inx2 = 3 * E;
	//TODO: load from smooth initial:
	double Tdone = 0.135;
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
	
	if (upper >= 2 & lower <= 2)
	{
		//Logger::WriteMessage("level2");
		Common::Logger("level2");
		vector<string> level2 = { };
		{
			string key = "3";
			Interpolation i;
			i.interpolateGeneric(key, coef, tsec, na, d, inx1, inx2, r, sigma, T, E, level2, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			string key = "2";
			Interpolation i;
			i.interpolateGeneric(key, coef, tsec, nb, d, inx1, inx2, r, sigma, T, E, level2, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	if (upper >= 3 & lower <= 3)
	{
		//Level 3 ....multilevel method has to use all previous information
		//Logger::WriteMessage("level3");
		Common::Logger("level3");

		vector<string> level3 = { "2","3" };
		{
			Interpolation i;
			string key = "4";
			i.interpolateGeneric(key, coef, tsec, na + 1, d, inx1, inx2, r, sigma, T, E, level3, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_3";
			i.interpolateGeneric(key, coef, tsec, nb + 1, d, inx1, inx2, r, sigma, T, E, level3, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	if (upper >= 4 & lower <= 4)
	{
		//Level 4 ....higher level needs more information
		//Logger::WriteMessage("level4");
		Common::Logger("level4");
		vector<string> level4 = { "2","3","_3","4" };
		{
			Interpolation i;
			string key = "5";
			i.interpolateGeneric(key, coef, tsec, na + 2, d, inx1, inx2, r, sigma, T, E, level4, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_4";
			i.interpolateGeneric(key, coef, tsec, nb + 2, d, inx1, inx2, r, sigma, T, E, level4, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	if (upper >= 5 & lower <= 5)
	{
		//Level 5
		//Logger::WriteMessage("level5");
		Common::Logger("level5");
		vector<string> level5 = { "2","3","_3","4","_4","5" };
		{
			Interpolation i;
			string key = "6";
			i.interpolateGeneric(key, coef, tsec, na + 3, d, inx1, inx2, r, sigma, T, E, level5, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_5";
			i.interpolateGeneric(key, coef, tsec, nb + 3, d, inx1, inx2, r, sigma, T, E, level5, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	if (upper >= 6 & lower <= 6)
	{
		//Level 6
		//Logger::WriteMessage("level6");
		Common::Logger("level6");
		vector<string> level6 = { "2","3","_3","4","_4","5", "_5", "6" };
		{
			Interpolation i;
			string key = "7";
			i.interpolateGeneric(key, coef, tsec, na + 4, d, inx1, inx2, r, sigma, T, E, level6, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_6";
			i.interpolateGeneric(key, coef, tsec, nb + 4, d, inx1, inx2, r, sigma, T, E, level6, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	if (upper >= 7 & lower <= 7)
	{
		//Level7
		//Logger::WriteMessage("level7");
		Common::Logger("level7");
		vector<string> level7 = { "2","3","_3","4","_4","5","_5","6", "_6", "7" };
		{
			Interpolation i;
			string key = "8";
			i.interpolateGeneric(key, coef, tsec, na + 5, d, inx1, inx2, r, sigma, T, E, level7, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_7";
			i.interpolateGeneric(key, coef, tsec, nb + 5, d, inx1, inx2, r, sigma, T, E, level7, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	if (upper >= 8 & lower <= 8)
	{
		//Level8
		//Logger::WriteMessage("level8");
		Common::Logger("level8");
		vector<string> level8 = { "2","3","_3","4","_4","5","_5","6", "_6", "7", "_7","8" };
		{
			Interpolation i;
			string key = "9";
			i.interpolateGeneric(key, coef, tsec, na + 6, d, inx1, inx2, r, sigma, T, E, level8, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_8";
			i.interpolateGeneric(key, coef, tsec, nb + 6, d, inx1, inx2, r, sigma, T, E, level8, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	if (upper >= 9 & lower <= 9)
	{
		//Level9
		//Logger::WriteMessage("level9");
		Common::Logger("level9");
		vector<string> level9 = { "2","3","_3","4","_4","5","_5","6", "_6", "7", "_7","8", "_8","9" };
		{
			Interpolation i;
			string key = "10";
			i.interpolateGeneric(key, coef, tsec, na + 7, d, inx1, inx2, r, sigma, T, E, level9, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_9";
			i.interpolateGeneric(key, coef, tsec, nb + 7, d, inx1, inx2, r, sigma, T, E, level9, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	if (upper >= 10 & lower <= 10)
	{
		//Level10
		//Logger::WriteMessage("level10");
		Common::Logger("level10");
		vector<string> level10 = { "2","3","_3","4","_4","5","_5","6", "_6", "7", "_7","8", "_8","9", "_9","10" };
		{
			Interpolation i;
			string key = "11";
			i.interpolateGeneric(key, coef, tsec, na + 8, d, inx1, inx2, r, sigma, T, E, level10, &interpolation);
			interpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_10";
			i.interpolateGeneric("_10", coef, tsec, nb + 8, d, inx1, inx2, r, sigma, T, E, level10, &interpolation);
			interpolation[key] = i.getResult();
		}
	}
	//Logger::WriteMessage("inter_test");
	Common::Logger("inter_test");

	
	//interTest.Execute(vInterpolation, TX);
	InterTest interTest;
	vector<thread> threads;
	vector<InterTest> interTests;
		
	VectorXd U = VectorXd::Zero(10000);
	if (upper >= 2 & lower <= 2)
	{
		InterTest interTest2;
		InterTest interTest3;
		
		vector<vector<MatrixXd>> test2 = interpolation["2"];
		threads.push_back(std::thread(&InterTest::parallel, interTest2, "2", TX, test2[0], test2[1], test2[2], test2[3]));
		//VectorXd V_2 = interTest.serial(TX, test2[0], test2[1], test2[2], test2[3]);
		vector<vector<MatrixXd>> test3 = interpolation["3"];
		threads.push_back(std::thread(&InterTest::parallel, interTest3, "3", TX, test3[0], test3[1], test3[2], test3[3]));
		//VectorXd V3 = interTest.serial(TX, test3[0], test3[1], test3[2], test3[3]);
		//U = V3 - V_2;
		//uMatrix[0] = U;
		interTests.push_back(interTest2);
		interTests.push_back(interTest3);
	}
	VectorXd U1 = VectorXd::Zero(10000);
	if (upper >= 3 & lower <= 3)
	{
		InterTest interTest_3;
		InterTest interTest4;
		vector<vector<MatrixXd>> test_3 = interpolation["_3"];
		threads.push_back(std::thread(&InterTest::parallel, interTest_3, "_3", TX, test_3[0], test_3[1], test_3[2], test_3[3]));
		//VectorXd V_3 = interTest.serial(TX, test_3[0], test_3[1], test_3[2], test_3[3]);
		vector<vector<MatrixXd>> test4 = interpolation["4"];
		threads.push_back(std::thread(&InterTest::parallel, interTest4, "4", TX, test4[0], test4[1], test4[2], test4[3]));
		//VectorXd V4 = interTest.serial(TX, test4[0], test4[1], test4[2], test4[3]);
		//U1 = V4 - V_3;
		//uMatrix[1] = U1;
		interTests.push_back(interTest_3);
		interTests.push_back(interTest4);
	}
	VectorXd U2 = VectorXd::Zero(10000);
	if (upper >= 4 & lower <= 4)
	{
		InterTest interTest_4;
		InterTest interTest5;
		vector<vector<MatrixXd>> test_4 = interpolation["_4"];
		threads.push_back(std::thread(&InterTest::parallel, interTest_4, "_4", TX, test_4[0], test_4[1], test_4[2], test_4[3]));
		//VectorXd V_4 = interTest.serial(TX, test_4[0], test_4[1], test_4[2], test_4[3]);
		vector<vector<MatrixXd>> test5 = interpolation["5"];
		threads.push_back(std::thread(&InterTest::parallel, interTest5, "5", TX, test5[0], test5[1], test5[2], test5[3]));
		//VectorXd V5 = interTest.serial(TX, test5[0], test5[1], test5[2], test5[3]);
		//U2 = V5 - V_4;
		//uMatrix[2] = U2;
		interTests.push_back(interTest_4);
		interTests.push_back(interTest5);
	}
	VectorXd U3 = VectorXd::Zero(10000);
	if (upper >= 5 & lower <= 5)
	{
		InterTest interTest_5;
		InterTest interTest6;
		vector<vector<MatrixXd>> test_5 = interpolation["_5"];
		threads.push_back(std::thread(&InterTest::parallel, interTest_5, "_5", TX, test_5[0], test_5[1], test_5[2], test_5[3]));
		//VectorXd V_5 = interTest.serial(TX, test_5[0], test_5[1], test_5[2], test_5[3]);
		vector<vector<MatrixXd>> test6 = interpolation["6"];
		threads.push_back(std::thread(&InterTest::parallel, interTest6, "6", TX, test6[0], test6[1], test6[2], test6[3]));
		//VectorXd V6 = interTest.serial(TX, test6[0], test6[1], test6[2], test6[3]);
		//U3 = V6 - V_5;
		//uMatrix[3] = U3;
		interTests.push_back(interTest_5);
		interTests.push_back(interTest6);
	}
	VectorXd U4 = VectorXd::Zero(10000);
	if (upper >= 6 & lower <= 6)
	{
		InterTest interTest_6;
		InterTest interTest7;
		vector<vector<MatrixXd>> test_6 = interpolation["_6"];
		threads.push_back(std::thread(&InterTest::parallel, interTest_6, "_6", TX, test_6[0], test_6[1], test_6[2], test_6[3]));
		//VectorXd V_6 = interTest.serial(TX, test_6[0], test_6[1], test_6[2], test_6[3]);
		vector<vector<MatrixXd>> test7 = interpolation["7"];
		threads.push_back(std::thread(&InterTest::parallel, interTest7, "7", TX, test7[0], test7[1], test7[2], test7[3]));
		//VectorXd V7 = interTest.serial(TX, test7[0], test7[1], test7[2], test7[3]);
		//U4 = V7 - V_6;
		//uMatrix[4] = U4;
		interTests.push_back(interTest_6);
		interTests.push_back(interTest7);
	}
	VectorXd U5 = VectorXd::Zero(10000);
	if (upper >= 7 & lower <= 7)
	{
		InterTest interTest_7;
		InterTest interTest8;
		vector<vector<MatrixXd>> test_7 = interpolation["_7"];
		threads.push_back(std::thread(&InterTest::parallel, interTest_7, "_7", TX, test_7[0], test_7[1], test_7[2], test_7[3]));
		//VectorXd V_7 = interTest.serial(TX, test_7[0], test_7[1], test_7[2], test_7[3]);
		vector<vector<MatrixXd>> test8 = interpolation["8"];
		threads.push_back(std::thread(&InterTest::parallel, interTest8, "8", TX, test8[0], test8[1], test8[2], test8[3]));
		//VectorXd V8 = interTest.serial(TX, test8[0], test8[1], test8[2], test8[3]);
		//U5 = V8 - V_7;
		//uMatrix[5] = U5;
		interTests.push_back(interTest_7);
		interTests.push_back(interTest8);
	}
	VectorXd U6 = VectorXd::Zero(10000);
	if (upper >= 8 & lower <= 8)
	{
		InterTest interTest_8;
		InterTest interTest9;
		vector<vector<MatrixXd>> test_8 = interpolation["_8"];
		threads.push_back(std::thread(&InterTest::parallel, interTest_8, "_8", TX, test_8[0], test_8[1], test_8[2], test_8[3]));
		//VectorXd V_8 = interTest.serial(TX, test_8[0], test_8[1], test_8[2], test_8[3]);
		vector<vector<MatrixXd>> test9 = interpolation["9"];
		threads.push_back(std::thread(&InterTest::parallel, interTest9, "9", TX, test9[0], test9[1], test9[2], test9[3]));
		//VectorXd V9 = interTest.serial(TX, test9[0], test9[1], test9[2], test9[3]);
		//U6 = V9 - V_8;
		//uMatrix[6] = U6;
		interTests.push_back(interTest_8);
		interTests.push_back(interTest9);
	}
	VectorXd U7 = VectorXd::Zero(10000);
	if (upper >= 9 & lower <= 9)
	{
		InterTest interTest_9;
		InterTest interTest10;
		vector<vector<MatrixXd>> test_9 = interpolation["_9"];
		threads.push_back(std::thread(&InterTest::parallel, interTest_9, "_9", TX, test_9[0], test_9[1], test_9[2], test_9[3]));
		//VectorXd V_9 = interTest.serial(TX, test_9[0], test_9[1], test_9[2], test_9[3]);
		vector<vector<MatrixXd>> test10 = interpolation["10"];
		//VectorXd V10 = interTest.serial(TX, test10[0], test10[1], test10[2], test10[3]);
		threads.push_back(std::thread(&InterTest::parallel, interTest10, "10", TX, test10[0], test10[1], test10[2], test10[3]));
		//U7 = V10 - V_9;
		//uMatrix[7] = U7;
		interTests.push_back(interTest_9);
		interTests.push_back(interTest10);
	}
	VectorXd U8 = VectorXd::Zero(10000);
	if (upper >= 10 & lower <= 10)
	{
		InterTest interTest_10;
		InterTest interTest11;
		vector<vector<MatrixXd>> test_10 = interpolation["_10"];
		threads.push_back(std::thread(&InterTest::parallel, interTest_10, "_10", TX, test_10[0], test_10[1], test_10[2], test_10[3]));
		//VectorXd V_10 = interTest.serial(TX, test_10[0], test_10[1], test_10[2], test_10[3]);
		vector<vector<MatrixXd>> test11 = interpolation["11"];
		threads.push_back(std::thread(&InterTest::parallel, interTest11, "11", TX, test11[0], test11[1], test11[2], test11[3]));
		//VectorXd V11 = interTest.serial(TX, test11[0], test11[1], test11[2], test11[3]);
		//U8 = V11 - V_10;
		//uMatrix[8] = U8;
		interTests.push_back(interTest_10);
		interTests.push_back(interTest11);
	}

	for (int i = 0; i < threads.size(); i++)
		threads.at(i).join();

	//Logger::WriteMessage("inter_test complete");
	Common::Logger("inter_test complete");

	if (upper >= 2 & lower <= 2)
		U = interTests.at(1).GetResult("3") - interTests.at(0).GetResult("2");
	if (upper >= 3 & lower <= 3)
		U1 = interTests.at(3).GetResult("4") - interTests.at(2).GetResult("_3");
	if (upper >= 4 & lower <= 4)
		U2 = interTests.at(5).GetResult("5") - interTests.at(4).GetResult("_4");
	if (upper >= 5 & lower <= 5)
		U3 = interTests.at(7).GetResult("6") - interTests.at(6).GetResult("_5");
	if (upper >= 6 & lower <= 6)
		U4 = interTests.at(9).GetResult("7") - interTests.at(8).GetResult("_6");
	if (upper >= 7 & lower <= 7)
		U5 = interTests.at(11).GetResult("8") - interTests.at(10).GetResult("_7");
	if (upper >= 8 & lower <= 8)
		U6 = interTests.at(13).GetResult("9") - interTests.at(12).GetResult("_8");
	if (upper >= 9 & lower <= 9)
		U7 = interTests.at(15).GetResult("10") - interTests.at(14).GetResult("_9");
	if (upper >= 10 & lower <= 10)
		U8 = interTests.at(17).GetResult("11") - interTests.at(16).GetResult("_10");

	//[AP] = ECP(TX, r, sigma, T, E);
	VectorXd AP = ECP(TX, r, sigma, T, E);
	//Logger::WriteMessage("MuSik addition");
	Common::Logger("MuSIK addition");
	int m = U.rows();
	MatrixXd MuSIK = MatrixXd::Zero(m,9);
	if (upper >= 2 & lower <= 2)
		MuSIK.col(0) = U;
	if (upper >= 3 & lower <= 3)
		MuSIK.col(1) = U + U1;
	if (upper >= 4 & lower <= 4)
		MuSIK.col(2) = U + U1 + U2;
	if (upper >= 5 & lower <= 5)
		MuSIK.col(3) = U + U1 + U2 + U3;
	if (upper >= 6 & lower <= 6)
		MuSIK.col(4) = U + U1 + U2 + U3 + U4;
	if (upper >= 7 & lower <= 7)
		MuSIK.col(5) = U + U1 + U2 + U3 + U4 + U5;
	if (upper >= 8 & lower <= 8)
		MuSIK.col(6) = U + U1 + U2 + U3 + U4 + U5 + U6;
	if (upper >= 9 & lower <= 9)
		MuSIK.col(7) = U + U1 + U2 + U3 + U4 + U5 + U6 + U7;
	if (upper >= 10 & lower <= 10)
		MuSIK.col(8) = U + U1 + U2 + U3 + U4 + U5 + U6 + U7 + U8;

	VectorXd RMS = VectorXd::Ones(9,1);
	VectorXd Max = VectorXd::Ones(9, 1);

	//Logger::WriteMessage("RMS calcs");
	Common::Logger("RMD calcs");
	for (int i = 0; i < MuSIK.cols(); i++)
	{
		VectorXd v = MuSIK.col(i).array() - AP.array();
		RMS[i] = RootMeanSquare(v);
		VectorXd m = abs(MuSIK.col(i).array() - AP.array());
		Max[i] = m.maxCoeff();
	}

	vInterpolation = interpolation;
	vector<MatrixXd> result = { MuSIK, RMS, Max };
	return result;
}
double SparseGridCollocation::RootMeanSquare(VectorXd v)
{
	double rms = sqrt((v.array() * v.array()).sum() / v.size() );
	return rms;
}

