// SparseGridGollocation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialX.h"
#include "SmoothInitialU.h"
#include "SparseGridCollocation.h"
#include "windows.h"
#include "Common.h"
#include "Interpolation.h"
#include "RBF.h"
#include "InterTest.h"
#include <iomanip>
#include "Params.h"
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

vector<MatrixXd> SparseGridCollocation::MuSIKc(int upper, int lower, Params p)
{
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
	

	vector<VectorXd> smoothinitial = MethodOfLines(p);
	//HACK: last element of smooth initial is incorrect:
	//wcout << Common::printMatrix(smoothinitial[0]) << endl;
	//wcout << Common::printMatrix(smoothinitial[1]) << endl;
	//SmoothInitialX::x = smoothinitial[0].head(smoothinitial[0].rows() - 2);
	SmoothInitialX::x = smoothinitial[0].head(smoothinitial[0].rows());
	//SmoothInitialU::u = smoothinitial[1].head(smoothinitial[1].rows() - 2);
	SmoothInitialU::u = smoothinitial[1].head(smoothinitial[1].rows());

	wcout << Common::printMatrix(SmoothInitialX::x) << endl;
	wcout << Common::printMatrix(SmoothInitialU::u) << endl;

	p.Tdone = 0.1350;
	p.inx1 = 0;
	p.inx2 = 3.0 * p.K;

	return MuSIKc(upper, lower, p, vInterpolation);
}

vector<MatrixXd> SparseGridCollocation::MuSIKc(int upper, int lower, Params p, map<string, vector<vector<MatrixXd>>>& interpolation)
{
	string testPath = "C:\\Users\\User\\Documents\\Dissertation\\Matlab\\sparse collocation\\Black Scholes\\1 Asset\\sparse grid collocation";
	cout << "Starting MuSiK-c" << endl;

	double E = p.K;// strike price

	double r = p.r; // interest rate
	//double sigma = 0.15;
	double sigma = p.sigma;
	double T = p.T; // Maturity
	//double inx1 = 0; // stock price S belongs to[inx1 inx2]
	double inx1 = p.inx1;
	//double inx2 = 3 * E;
	double inx2 = p.inx2;
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
	{
		MatrixXd V3 = interTests.at(1).GetResult("3");
		MatrixXd V_2 = interTests.at(0).GetResult("2");
		U = V3 - V_2;
		
		MatrixXd v_2 = Common::ReadBinary(testPath, "V2.txt", V_2.rows(), V_2.cols());
		Common::checkMatrix(v_2, V_2, DBL_EPSILON, false);
		MatrixXd v3 = Common::ReadBinary(testPath, "V3.txt", V_2.rows(), V_2.cols());
		Common::checkMatrix(v3, V3, DBL_EPSILON, false);
	}
	if (upper >= 3 & lower <= 3)
	{
		MatrixXd V4 = interTests.at(3).GetResult("4");
		MatrixXd V_3 = interTests.at(2).GetResult("_3");
		U1 = V4 - V_3;
		MatrixXd v_3 = Common::ReadBinary(testPath, "V_3.txt", V_3.rows(), V_3.cols());
		Common::checkMatrix(v_3, V_3, DBL_EPSILON, false);
		MatrixXd v4 = Common::ReadBinary(testPath, "V4.txt", V_3.rows(), V_3.cols());
		Common::checkMatrix(v4, V4, DBL_EPSILON, false);
	}
	if (upper >= 4 & lower <= 4)
	{
		MatrixXd V_4 = interTests.at(4).GetResult("_4");
		MatrixXd V5 = interTests.at(5).GetResult("5");
		U2 =  V5 - V_4;
		MatrixXd v_4 = Common::ReadBinary(testPath, "V_4.txt", V_4.rows(), V_4.cols());
		Common::checkMatrix(v_4, V_4, DBL_EPSILON, false);
		MatrixXd v5 = Common::ReadBinary(testPath, "V5.txt", V_4.rows(), V_4.cols());
		Common::checkMatrix(v5, V5, DBL_EPSILON, false);
	}
	if (upper >= 5 & lower <= 5)
	{
		MatrixXd V_5 = interTests.at(6).GetResult("_5");
		MatrixXd V6 = interTests.at(7).GetResult("6");
		U3 = V6 - V_5;
		MatrixXd v_5 = Common::ReadBinary(testPath, "V_5.txt", V_5.rows(), V_5.cols());
		Common::checkMatrix(v_5, V_5, DBL_EPSILON, false);
		MatrixXd v6 = Common::ReadBinary(testPath, "V6.txt", V_5.rows(), V_5.cols());
		Common::checkMatrix(v6, V6, DBL_EPSILON, false);
	}
	if (upper >= 6 & lower <= 6)
	{
		MatrixXd V_6 = interTests.at(8).GetResult("_6");
		MatrixXd V7 = interTests.at(9).GetResult("7");
		U4 = V7 - V_6;
		MatrixXd v_6 = Common::ReadBinary(testPath, "V_6.txt", V_6.rows(), V_6.cols());
		Common::checkMatrix(v_6, V_6, DBL_EPSILON, false);
		MatrixXd v7 = Common::ReadBinary(testPath, "V7.txt", V_6.rows(), V_6.cols());
		Common::checkMatrix(v7, V7, DBL_EPSILON, false);
	}
	if (upper >= 7 & lower <= 7)
	{
		MatrixXd V_7 = interTests.at(10).GetResult("_7");
		MatrixXd V8 = interTests.at(11).GetResult("8");
		U5 = V8 - V_7;
		MatrixXd v_7 = Common::ReadBinary(testPath, "V_7.txt", V_7.rows(), V_7.cols());
		Common::checkMatrix(v_7, V_7, DBL_EPSILON, false);
		MatrixXd v8 = Common::ReadBinary(testPath, "V8.txt", V_7.rows(), V_7.cols());
		Common::checkMatrix(v8, V8, DBL_EPSILON, false);
	}
	if (upper >= 8 & lower <= 8)
	{
		MatrixXd V_8 = interTests.at(12).GetResult("_8");
		MatrixXd V9 = interTests.at(13).GetResult("9");
		U6 = V9 - V_8;
		MatrixXd v_8 = Common::ReadBinary(testPath, "V_8.txt", V_8.rows(), V_8.cols());
		Common::checkMatrix(v_8, V_8, DBL_EPSILON, false);
		MatrixXd v9 = Common::ReadBinary(testPath, "V9.txt", V_8.rows(), V_8.cols());
		Common::checkMatrix(v9, V9, DBL_EPSILON, false);
	}
	if (upper >= 9 & lower <= 9)
	{
		MatrixXd V_9 = interTests.at(14).GetResult("_9");
		MatrixXd V10 = interTests.at(15).GetResult("10");
		U7 = V10 - V_9;
		MatrixXd v_9 = Common::ReadBinary(testPath, "V_9.txt", V_9.rows(), V_9.cols());
		Common::checkMatrix(v_9, V_9, DBL_EPSILON, false);
		MatrixXd v10 = Common::ReadBinary(testPath, "V10.txt", V_9.rows(), V_9.cols());
		Common::checkMatrix(v10, V10, DBL_EPSILON, false);
	}
	if (upper >= 10 & lower <= 10)
	{
		MatrixXd V_10 = interTests.at(16).GetResult("_10");
		MatrixXd V11 = interTests.at(17).GetResult("11");
		U8 = V11 - V_10;
		MatrixXd v_10 = Common::ReadBinary(testPath, "V_10.txt", V_10.rows(), V_10.cols());
		Common::checkMatrix(v_10, V_10, DBL_EPSILON, false);
		MatrixXd v11 = Common::ReadBinary(testPath, "V11.txt", V11.rows(), V11.cols());
		Common::checkMatrix(v11, V11, DBL_EPSILON, false);
	}
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

VectorXd PayOffFunction(VectorXd S, double K)
{
	VectorXd delta = S.array() - K;
	VectorXd result = (delta.array() > 0).select(delta, 0);
	return result;
}

VectorXd Diff(VectorXd A)
{
	VectorXd result(A.rows() - 1);
	for (int i = 0; i < A.rows() - 1; i++)
	{
		result[i] = A[i+1] - A[i];
	}
	return result;

}

VectorXd Push(VectorXd A, double push)
{
	VectorXd result(A.rows() + 1);
	result[0] = push;
	for (int i = 1; i <= A.rows(); i++)
	{
		result[i] = A[i-1];
	}
	
	return result;

}

VectorXd Queue(VectorXd A, double queue)
{
	VectorXd result(A.rows() + 1);

	for (int i = 0; i < A.rows(); i++)
	{
		result[i] = A[i];
	}
	result[A.rows()] = queue;
	return result;
}

VectorXd PushAndQueue(double push, VectorXd A, double queue)
{
	VectorXd result(A.rows() + 2);
	result[0] = push;
	for (int i = 0; i <= A.rows(); i++)
	{
		result[i+1] = A[i];
	}
	result[A.rows() + 1] = queue;
	return result;
}

VectorXd SparseGridCollocation::Select(VectorXd A, double notEqual)
{
	vector<double> inter;
	for (int i = 0; i < A.rows(); i++)
	{
		if (A[i] != notEqual)
			inter.push_back(A[i]);
	}
	VectorXd result(inter.size());
	int count = 0;
	for (auto i : inter)
	{
		result[count] = i;
		count++;
	}
	return result;
}

vector<VectorXd> SparseGridCollocation::MethodOfLines(Params p)
{
	return MethodOfLines(p.T, p.Tdone, p.Tend, p.dt, p.K, p.r, p.sigma, p.theta, p.inx1, p.inx2);
}

vector<VectorXd> SparseGridCollocation::MethodOfLines(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2)
{
	cout << "MethodOfLines for 1-D European Call Option" << endl;
	vector<VectorXd> price = EuroCallOption1D(T, Tdone, Tend, dt, K, r, sigma, theta, inx1, inx2);
	//wcout << "price x:" << endl;
	//wcout << Common::printMatrix(price[0]) << endl;
	//Common::saveArray(price[0], "x.txt");
	//wcout << "price lamb:" << endl;
	//wcout << Common::printMatrix(price[1]) << endl;
	//Common::saveArray(price[1], "lamb.txt");
	//wcout << "price c:" << endl;
	//wcout << Common::printMatrix(price[2]) << endl;
	//Common::saveArray(price[2], "c.txt");

	VectorXd x = price[0];
	VectorXd lamb = price[1];
	VectorXd c = price[2];
	double Smin = 0;
	double Smax = 3 * K;
	double twoPower15Plus1 = pow(2, 15) + 1;// 32768; //pow(2, 15) + 1
	//wcout << "twoPower15Plus1=" << twoPower15Plus1 << endl;
	VectorXd X_ini = VectorXd::LinSpaced(twoPower15Plus1, Smin, Smax);
	
	MatrixXd phi = MatrixXd::Ones(X_ini.size(), x.size());
	for (int i = 0; i < x.size(); i++)
	{
		vector<MatrixXd> rbf = RBF::mqd1(X_ini, x(i), c(i));
		phi.col(i) = rbf[0].col(0);
	}

	//string testPath = "C:\\Users\\User\\Documents\\Dissertation\\Matlab\\sparse collocation";
	//MatrixXd uX_ini = Common::ReadBinary(testPath, "X_ini.txt", 5000, 1);
	//cout << "X ini error" << endl;
	//Common::checkMatrix(uX_ini, X_ini, DBL_EPSILON, false);
	//MatrixXd uPhi = Common::ReadBinary(testPath, "phi.txt", 5000, 1);
	//cout << "phi error" << endl;
	//Common::checkMatrix(uPhi, phi, DBL_EPSILON, false);
	//MatrixXd ulamb = Common::ReadBinary(testPath, "lamb.final.txt", 5000, 1);
	//cout << "lamb error" << endl;
	//Common::checkMatrix(ulamb, lamb, DBL_EPSILON, false);

	VectorXd U_ini = phi * lamb;
	//MatrixXd uU_ini = Common::ReadBinary(testPath, "U_ini.txt", 5000, 1);
	//cout << "U ini error" << endl;
	//Common::checkMatrix(uU_ini, U_ini, DBL_EPSILON, false);

	//Common::saveArray(X_ini, "xini.txt");
	//Common::saveArray(U_ini, "uini.txt");
	return { X_ini, U_ini };
}

vector<VectorXd> SparseGridCollocation::EuroCallOption1D(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2)
{
	string testPath = "C:\\Users\\User\\Documents\\Dissertation\\Matlab\\sparse collocation";
	int N_uniform = 5000;
	VectorXd x = VectorXd::LinSpaced(N_uniform, inx1, inx2);
	//MatrixXd uX = Common::ReadBinary(testPath, "x.txt", 5000, 1);
	//Common::checkMatrix(uX, x, DBL_EPSILON, false);
	//cout << "x linspaced test complete" << endl;
	//Common::saveArray(uX, "eX.txt");
	//Common::saveArray(uX, "aX.txt");

	VectorXd u0 = PayOffFunction(x, K);
	
	VectorXd dx = Diff(x);
	//dx[0] = dx[1];
	const double inf = numeric_limits<double>::infinity();
	VectorXd vInf(dx.rows());
	//vInf = VectorXd::fill(&inf);
	vInf.fill(inf);
	Common::saveArray(dx, "dx.txt");

	VectorXd pushed = Push(dx, inf);
	VectorXd queued = Queue(dx, inf);
	VectorXd c = 2 * (pushed.array() >= queued.array()).select(queued, pushed);
	//VectorXd c = 2 * (dx.array() <= vInf.array()).select(dx, vInf);
	//wcout << Common::printMatrix(dx) << endl;
	//wcout << Common::printMatrix(c) << endl;
	//Common::saveArray(pushed, "pushed.txt");
	//Common::saveArray(queued, "queued.txt");
	//Common::saveArray(c, "c.txt");

	VectorXd xx = VectorXd::LinSpaced(1000, 0, 3 * K);
	VectorXd IT = ( (1.2 * K >= x.array()) && (x.array() >= 0.8 * K) ).select(x, 0);
	double sumIT = IT.sum();
	VectorXd AroundE = Select(IT, 0);

	int N = x.size();
	MatrixXd D1_mid = MatrixXd::Zero(AroundE.size(), N);
	MatrixXd D2_mid = MatrixXd::Zero(AroundE.size(), N);
	MatrixXd D3_mid = MatrixXd::Zero(AroundE.size(), N);
	
	MatrixXd A = MatrixXd::Zero(N, N);
	MatrixXd D1 = MatrixXd::Zero(N, N);
	MatrixXd D2 = MatrixXd::Zero(N, N);

	MatrixXd Axx = MatrixXd::Zero(xx.size(), N);

	for (int j = 0; j < N; j++)
	{
		//wcout << setprecision(25) << c[j] << endl;
		vector<MatrixXd> vAxx =  RBF::mqd1(xx, x[j], c[j]);
		Axx.col(j) = vAxx[0].col(0);
		//wcout << Common::printMatrix(Axx.col(j)) << endl;
		cout << "RBF::mqd1 x[j] ="<< x[j] << " c[j]=" << c[j] << endl;
		vector<MatrixXd> vAx = RBF::mqd1(x, x[j], c[j]);
		A.col(j) = vAx[0].col(0);
		//stringstream ss;
		//ss << "Acol" << j << ".txt";
		//MatrixXd uA = Common::ReadBinary(testPath, ss.str(), 5000, 1);
		//Common::checkMatrix(uA, A.col(j), DBL_EPSILON, false);
		//cout << "Acol" << j << " test complete" << endl;
		//if (j >= 4990)
		//	Common::saveArray(A.col(j), "A.colj.txt");
		//wcout << Common::printMatrix(A.col(j)) << endl;
		D1.col(j) = vAx[1].col(0);
		D2.col(j) = vAx[2].col(0);
		vector<MatrixXd> vAE = RBF::mqd1(AroundE, x[j], c[j]);
		D1_mid.col(j) = vAE[1].col(0);
		D2_mid.col(j) = vAE[2].col(0);
		
		D3_mid.col(j) = vAE[3].col(0);
		D1_mid.col(j) = D1_mid.col(j).array() / AroundE.array();
		D2_mid.col(j) = D2_mid.col(j).array() / (AroundE.array() * AroundE.array());
		//wcout << Common::printMatrix(D3_mid.col(j)) << endl;
	}
	
	//Logger::WriteMessage(Common::printMatrix(u0).c_str());
	//Common::saveArray(A, "A.txt");
	//Common::saveArray(u0, "u0.txt");

	
	//MatrixXd uuu0 = Common::ReadBinary(testPath, "u0.txt", 5000, 1);
	//Common::checkMatrix(uuu0, u0, DBL_EPSILON, false);

	//MatrixXd uA = Common::ReadBinary(testPath, "A.txt", 5000, 1);
	//Common::checkMatrix(uA, A, DBL_EPSILON, false);

	VectorXd lamb = A.lu().solve(u0);
	/*MatrixXd uLamb = Common::ReadBinary(testPath, "lamb.txt", 5000, 1);
	Common::checkMatrix(uLamb, lamb, DBL_EPSILON, false);
	VectorXd lamb1 = A.fullPivLu().solve(u0);
	Common::checkMatrix(uLamb, lamb1, DBL_EPSILON, false);
	VectorXd lamb2 = A.householderQr().solve(u0);
	Common::checkMatrix(uLamb, lamb2, DBL_EPSILON, false);
	VectorXd lamb3 = A.colPivHouseholderQr().solve(u0);
	Common::checkMatrix(uLamb, lamb3, DBL_EPSILON, false);
	VectorXd lamb4 = A.fullPivHouseholderQr().solve(u0);
	Common::checkMatrix(uLamb, lamb4, DBL_EPSILON, false);
	VectorXd lamb5 = A.llt().solve(u0);
	Common::checkMatrix(uLamb, lamb5, DBL_EPSILON, false);
	VectorXd lamb6 = A.ldlt().solve(u0);
	Common::checkMatrix(uLamb, lamb6, DBL_EPSILON, false);*/

	//Common::saveArray(lamb, "lamb.txt");
	//Common::saveArray(A, "A.txt");
	//Common::saveArray(D1, "D1.txt");
	//Common::saveArray(D2, "D2.txt");
	//wcout << Common::printMatrix(u0) << endl;

	//LLT<MatrixXd> lltOfA(A);
	//VectorXd lamb = lltOfA.solve(u0);
	//Common::saveArray(lamb, "lamb.txt");
	//wcout << Common::printMatrix(lamb) << endl;

	//wcout << Common::printMatrix(D2_mid) << endl;
	MatrixXd uu0 = Axx*lamb;
	MatrixXd deri1 = D1_mid*lamb;
	MatrixXd deri2 = D2_mid*lamb;
	//wcout << Common::printMatrix(lamb) << endl;
	//Common::saveArray(lamb, "lamb.txt");
	//wcout << Common::printMatrix(D2_mid) << endl;
	//Common::saveArray(D2_mid, "D2_mid.txt");
	
	MatrixXd deri3 = D3_mid*lamb;
	//Common::saveArray(deri3, "deri3.txt");

	MatrixXd A1 = A.row(1);
	MatrixXd Aend = A.row(A.rows() - 1);
	MatrixXd a = A.block(1, 0, A.rows() - 2, A.cols());
	//wcout << Common::printMatrix(a) << endl;
	MatrixXd d1 = D1.block(1, 0, D1.rows() - 2, D1.cols());
	//wcout << Common::printMatrix(d1) << endl;
	MatrixXd d2 = D2.block(1, 0, D2.rows() - 2, D2.cols());
	//wcout << Common::printMatrix(d2) << endl;

	//[Price] = ECP([ones(length(xx), 1).*(T - Tdone), xx], r, sig, T, E);
	//Common::saveArray(a, "a.txt");
	//Common::saveArray(d2, "d2.txt");
	//Common::saveArray(d1, "d1.txt");
	MatrixXd P = a * r - 0.5 * (sigma * sigma) * d2 - r * d1;
	MatrixXd H = a + dt * (1 - theta)* P;
	MatrixXd G = a - dt * theta * P;
	//wcout << Common::printMatrix(G) << endl;
	//Common::saveArray(P, "P.txt");

	int count = 0;
	cout << "MoL Iterative solver\r\n";
	//MethodOfLines::MoLiteration(Tend, Tdone, dt, G.data(), G.rows(), G.cols(), lamb.data(), lamb.rows(), lamb.cols(), inx2, r, K, A1, Aend, H);
	while (Tend - Tdone > 1E-8)
	{
		Tdone += dt;
		MatrixXd g = G* lamb;
		//Common::saveArray(g, "g.txt");
		VectorXd fff = PushAndQueue(0, g, inx2 - exp(-r*Tdone)*K);
		//Common::saveArray(fff, "fff.txt");
		//wcout << Common::printMatrix(fff) << endl;
		MatrixXd HH(A1.cols(), A1.cols());
		HH.row(0) = A1;
		HH.middleRows(1, HH.rows() -2) = H;
		HH.row(HH.rows() -1) = Aend;
		//Common::saveArray(HH, "HH.txt");
		//wcout << Common::printMatrix(HH) << endl;
		//LLT<MatrixXd> lltOfHH(HH);
		//lamb = lltOfHH.solve(fff);
		lamb = HH.lu().solve(fff);
		//stringstream ss;
		//ss << "lamb" << count << ".txt";
		//MatrixXd uLamb = Common::ReadBinary(testPath, ss.str(), 5000, 1);
		//Common::checkMatrix(uLamb, lamb, DBL_EPSILON, false);
		//stringstream ss;
		//ss << "lamb" << count << ".txt";
		//Common::saveArray(lamb, ss.str());
		//wcout << Common::printMatrix(lamb) << endl;

		MatrixXd uu0 = Axx*lamb;
		deri1 = D1_mid*lamb;
		deri2 = D2_mid*lamb;
		deri3 = D3_mid*lamb;
		//Common::saveArray(D3_mid, "D3_mid.txt");
		//Common::saveArray(lamb, "lamb.txt");
		//Common::saveArray(deri3, "deri3.txt");
		//Ptop = max(deri3); % peak top of Speed approximation
		double Ptop = deri3.maxCoeff();
		//	Pend = min(deri3); % peak end of Speed approximation
		double Pend = deri3.minCoeff();
		//	I1 = find(deri3 == Ptop);
		MatrixXd::Index I1Row, I1Col;
		deri3.maxCoeff(&I1Row, &I1Col);
		//I2 = find(deri3 == Pend);
		MatrixXd::Index I2Row, I2Col;
		deri3.minCoeff(&I2Row, &I2Col);
		//a = min(I1, I2);
		double part1Length = (I1Row < I2Row) ? I1Row : I2Row;
		//b = max(I1, I2);
		double part2Length = (I1Row > I2Row) ? I1Row : I2Row;
		//cout << "a:" << part1Length << "b:" << part2Length << endl;
		//part1 = diff(deri3(1:a));
		VectorXd vderi3 = deri3.col(0);
		MatrixXd part1 = Diff(vderi3.block(0, 0, part1Length, 1));
		//part2 = diff(deri3(a:b));
		MatrixXd prediff = vderi3.block(part1Length, 0, part2Length - part1Length + 1, 1);
		MatrixXd part2 = Diff(prediff);
		//cout << "part2 first:" << part2(0,0) << endl;
		//cout << "part2 2:" << part2(1, 0) << endl;
		//cout << "part2 3:" << part2(2, 0) << endl;
		//part3 = diff(deri3(b:end));
		MatrixXd part3 = Diff(vderi3.block(part2Length, 0, vderi3.rows() - part2Length, 1));
		//II1 = part1 >= 0;
		MatrixXd zeros = MatrixXd::Zero(part1.rows(), part1.cols());
		MatrixXd ones = MatrixXd::Ones(part1.rows(), part1.cols());
		MatrixXd II1 = (part1.array() >= zeros.array()).select(ones, zeros);
		//II2 = part2 >= 0;
		zeros = MatrixXd::Zero(part2.rows(), part2.cols());
		ones = MatrixXd::Ones(part2.rows(), part2.cols());
		MatrixXd II2 = (part2.array() >= zeros.array()).select(ones, zeros);
		//II3 = part3 >= 0;
		zeros = MatrixXd::Zero(part3.rows(), part3.cols());
		ones = MatrixXd::Ones(part3.rows(), part3.cols());
		MatrixXd II3 = (part3.array() >= zeros.array()).select(ones, zeros);

		double min = deri2.minCoeff();
		double II1sum = II1.sum();
		double II2sum = II2.sum();
		double II3sum = II3.sum();
		double p1size = part1.size();
		double p2size = part2.size();
		double p3size = part3.size();
		cout << "iteration:" << count << setprecision(8) << " T:" << Tdone << " minCoef:" << min;
		//if min(deri2) >= 0 % Gamma greater than 0
		if (min >= 0)
		{
			cout << " II1sum:" << II1sum << " part1 size:" << p1size;
			//	% Approximation of Speed is monotonic in subintervals
			//	if sum(II1) == 0 || sum(II1) == length(part1)
			if (II1.sum() == 0 || II1.sum() == part1.size())
			{
				cout << " II2sum:" << II2sum << " part2 size:" << p2size;
				//		if sum(II2) == 0 || sum(II2) == length(part2)
				//Common::saveArray(II2, "II2.txt");
				if (II2.sum() == 0 || II2.sum() == part2.size())
				{
					cout << " II3sum:" << II3sum << " part3 size:" << p3size;
					//			if sum(II3) == 0 || sum(II3) == length(part3)
					if (II3.sum() == 0 || II3.sum() == part3.size())
					{
						//				disp(Tdone)
						//				break
						//				end
						//				end
						//				end
						Common::saveArray(fff, "fff.txt");
						break;
					}
				}
			}
		}
		cout << "\r";
		count++;
		
	}
	cout << "Total Iterations:" << count << "\r\n";
	return { x, lamb, c };
	
	
}

