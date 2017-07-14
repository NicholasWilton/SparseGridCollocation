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



//using namespace Microsoft::VisualStudio::CppUnitTestFramework;

SparseGridCollocation::SparseGridCollocation()
{
}

map<string, vector<vector<MatrixXd>>> SparseGridCollocation::GetInterpolation()
{
	return vInterpolation;
}

MatrixXd SparseGridCollocation::ECP(MatrixXd X, double r, double sigma, double T, double E)
{
	// Analytical price for European call option
	//t = X(:, 1);
	VectorXd t= X.col(0);
	//S = X(:, 2);
	VectorXd S = X.col(1);
	//M = T - t;
	VectorXd M = T - t.array();
	//[N, ~] = size(X);
	int N = X.rows();
	//P = ones(N, 1);
	VectorXd P = VectorXd::Ones(N, 1);
	//d1 = ones(N, 1);
	VectorXd d1 = VectorXd::Ones(N, 1);
	//d2 = ones(N, 1);
	VectorXd d2 = VectorXd::Ones(N, 1);

	//I0 = M == 0;
	MatrixXd I0 = (M.array() == 0).select(M,0);
	//I1 = M ~= 0;
	MatrixXd I1 = (M.array() != 0).select(M,0);

	//P(I0) = max(S(I0) - E, 0);
	for (int i = 0; i < N; i++)
	{
		if (S(i) - E > 0)
			P(i) = S(i) - E;
		else
			P(i) = 0;


		//d1(I1) = (log(S(I1) . / E) + (r + sigma ^ 2 / 2).*M(I1)) . / (sigma.*sqrt(M(I1)));
		d1(i) = (log(S(i) / E) + (r + sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		//d2(I1) = (log(S(I1) . / E) + (r - sigma ^ 2 / 2).*M(I1)) . / (sigma.*sqrt(M(I1)));
		d2(i) = (log(S(i) / E) + (r - sigma * sigma / 2)* M(i)) / (sigma * sqrt(M(i)));
		//P(I1) = -E.*exp(-r.*M(I1)).*normcdf(d2(I1)) + S(I1).*normcdf(d1(I1));
		P(i) = S(i) * normCDF(d1(i)) - E * exp(-r * M(i)) * normCDF(d2(i));
	}
	return P;
}

double SparseGridCollocation::normCDF(double value)
{
	//return 0.5 * erfc(-value * M_SQRT1_2);
	return 0.5 * erfc(-value * (1/sqrt(2)));
}

double SparseGridCollocation::inner_test(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A )
{
	// This is used in the PDE system re - construct for initial and boundary conditions
	int ch = TX.size();
	vector<MatrixXd> V;
	//for j = 1:ch
	for (int j = 0; j < ch; j++)
	{
		//   multiquadric RBF......
		//     V1 = sqrt(((t - TX{ j }(:, 1))). ^ 2 + (C{ j }(1, 1). / A{ j }(1, 1)). ^ 2);
	//     V2 = sqrt(((x - TX{ j }(:, 2))). ^ 2 + (C{ j }(1, 2). / A{ j }(1, 2)). ^ 2);
	//     VV = V1.*V2;
	//     V(j) = VV'*lamb{j};
		//   .....................
		//   Gaussian RBF  .......
		//FAI1 = exp(-(A{ j }(1, 1)*(t - TX{ j }(:, 1))). ^ 2 / C{ j }(1, 1) ^ 2);
		MatrixXd square1 = (A[j](0, 0) * (t - (TX[j].col(0).array())).array()) * (A[j](0, 0) * (t - (TX[j].col(0).array())).array());
		MatrixXd FAI1 = (- square1 / (C[j](0, 0) * C[j](0, 0))).array().exp();

		//FAI2 = exp(-(A{ j }(1, 2)*(x - TX{ j }(:, 2))). ^ 2 / C{ j }(1, 2) ^ 2);
		MatrixXd square2 = (A[j](0, 1) * (t - (TX[j].col(1).array())).array()) * (A[j](0, 1) * (t - (TX[j].col(1).array())).array());
		MatrixXd FAI2 = (-square2 / (C[j](0, 1) * C[j](0, 1))).array().exp();
		//D = FAI1.*FAI2;
		VectorXd D = FAI1.cwiseProduct(FAI2).eval();
		//V(j) = D'*lamb{j};
		V.push_back(D.transpose() * lamb[j]);
			//   .....................
			//end
	}

		//output = sum(V);
	double output = 0;
	int i = 0;
	for (vector<MatrixXd>::iterator it = V.begin(); it < V.end(); it++, i++)
	{
		output += V[i].sum();

	}
	
	return output;
}

vector<VectorXd> SparseGridCollocation::MuSIKGeneric(int level)
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

	//t = linspace(0, tsec, N(1, 1));
	VectorXd x = VectorXd::LinSpaced(ch, inx1, inx2);
	VectorXd t = VectorXd::Zero(ch, 1);

	//TX = [t x']; //% testing nodes
	MatrixXd TX(ch, 2);
	TX << t, x;

	//% na, nb = level n + dimension d - 1
	int na = 3;
	int nb = 2;
	//tic


	// Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
	// C stands for shape parater, A stands for scale parameter
	//map<string, vector<vector<MatrixXd>> > vInterpolation;
	
	if (level >= 2)
	{
		//Logger::WriteMessage("level2");
		Common::Logger("level2");
		vector<string> level2 = { };
		{
			string key = "3";
			Interpolation i;
			i.interpolateGeneric(key, coef, tsec, na, d, inx1, inx2, r, sigma, T, E, level2, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			string key = "2";
			Interpolation i;
			i.interpolateGeneric(key, coef, tsec, nb, d, inx1, inx2, r, sigma, T, E, level2, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	if (level >= 3)
	{
		//Level 3 ....multilevel method has to use all previous information
		//Logger::WriteMessage("level3");
		Common::Logger("level3");

		vector<string> level3 = { "2","3" };
		{
			Interpolation i;
			string key = "4";
			i.interpolateGeneric(key, coef, tsec, na + 1, d, inx1, inx2, r, sigma, T, E, level3, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_3";
			i.interpolateGeneric(key, coef, tsec, nb + 1, d, inx1, inx2, r, sigma, T, E, level3, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	if (level >= 4)
	{
		//ttt(2) = toc;
		//Level 4 ....higher level needs more information
		//Logger::WriteMessage("level4");
		Common::Logger("level4");
		vector<string> level4 = { "2","3","_3","4" };
		{
			Interpolation i;
			string key = "5";
			i.interpolateGeneric(key, coef, tsec, na + 2, d, inx1, inx2, r, sigma, T, E, level4, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_4";
			i.interpolateGeneric(key, coef, tsec, nb + 2, d, inx1, inx2, r, sigma, T, E, level4, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	if (level >= 5)
	{
		//ttt(3) = toc;

		//Level 5
		//Logger::WriteMessage("level5");
		Common::Logger("level5");
		vector<string> level5 = { "2","3","_3","4","_4","5" };
		{
			Interpolation i;
			string key = "6";
			i.interpolateGeneric(key, coef, tsec, na + 3, d, inx1, inx2, r, sigma, T, E, level5, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_5";
			i.interpolateGeneric(key, coef, tsec, nb + 3, d, inx1, inx2, r, sigma, T, E, level5, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	if (level >= 6)
	{
		//ttt(4) = toc;

		//Level 6
		//Logger::WriteMessage("level6");
		Common::Logger("level6");
		vector<string> level6 = { "2","3","_3","4","_4","5", "_5", "6" };
		{
			Interpolation i;
			string key = "7";
			i.interpolateGeneric(key, coef, tsec, na + 4, d, inx1, inx2, r, sigma, T, E, level6, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_6";
			i.interpolateGeneric(key, coef, tsec, nb + 4, d, inx1, inx2, r, sigma, T, E, level6, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	if (level >= 7)
	{
		//ttt(5) = toc;

		//Level7
		//Logger::WriteMessage("level7");
		Common::Logger("level7");
		vector<string> level7 = { "2","3","_3","4","_4","5","_5","6", "_6", "7" };
		{
			Interpolation i;
			string key = "8";
			i.interpolateGeneric(key, coef, tsec, na + 5, d, inx1, inx2, r, sigma, T, E, level7, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_7";
			i.interpolateGeneric(key, coef, tsec, nb + 5, d, inx1, inx2, r, sigma, T, E, level7, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	if (level >= 8)
	{
		//Level8
		//Logger::WriteMessage("level8");
		Common::Logger("level8");
		vector<string> level8 = { "2","3","_3","4","_4","5","_5","6", "_6", "7", "_7","8" };
		{
			Interpolation i;
			string key = "9";
			i.interpolateGeneric(key, coef, tsec, na + 6, d, inx1, inx2, r, sigma, T, E, level8, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_8";
			i.interpolateGeneric(key, coef, tsec, nb + 6, d, inx1, inx2, r, sigma, T, E, level8, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	if (level >= 9)
	{
		//Level9
		//Logger::WriteMessage("level9");
		Common::Logger("level9");
		vector<string> level9 = { "2","3","_3","4","_4","5","_5","6", "_6", "7", "_7","8", "_8","9" };
		{
			Interpolation i;
			string key = "10";
			i.interpolateGeneric(key, coef, tsec, na + 7, d, inx1, inx2, r, sigma, T, E, level9, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_9";
			i.interpolateGeneric(key, coef, tsec, nb + 7, d, inx1, inx2, r, sigma, T, E, level9, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	if (level >= 10)
	{
		//Level10
		//Logger::WriteMessage("level10");
		Common::Logger("level10");
		vector<string> level10 = { "2","3","_3","4","_4","5","_5","6", "_6", "7", "_7","8", "_8","9", "_9","10" };
		{
			Interpolation i;
			string key = "11";
			i.interpolateGeneric(key, coef, tsec, na + 8, d, inx1, inx2, r, sigma, T, E, level10, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
		{
			Interpolation i;
			string key = "_10";
			i.interpolateGeneric("_10", coef, tsec, nb + 8, d, inx1, inx2, r, sigma, T, E, level10, &vInterpolation);
			vInterpolation[key] = i.getResult();
		}
	}
	//Logger::WriteMessage("inter_test");
	Common::Logger("inter_test");

	InterTest interTest;
	//interTest.Execute(vInterpolation, TX);
	
	VectorXd U = VectorXd::Zero(10000);
	if (level >= 2)
	{
		vector<vector<MatrixXd>> test2 = vInterpolation["2"];
		VectorXd V_2 = interTest.serial(TX, test2[0], test2[1], test2[2], test2[3]);
		vector<vector<MatrixXd>> test3 = vInterpolation["3"];
		VectorXd V3 = interTest.serial(TX, test3[0], test3[1], test3[2], test3[3]);
		U = V3 - V_2;
	}
	VectorXd U1 = VectorXd::Zero(10000);
	if (level >= 3)
	{
		vector<vector<MatrixXd>> test3 = vInterpolation["3"];
		vector<vector<MatrixXd>> test_3 = vInterpolation["_3"];
		VectorXd V_3 = interTest.serial(TX, test3[0], test3[1], test3[2], test3[3]);
		vector<vector<MatrixXd>> test4 = vInterpolation["4"];
		VectorXd V4 = interTest.serial(TX, test4[0], test4[1], test4[2], test4[3]);
		U1 = V4 - V_3;
	}
	VectorXd U2 = VectorXd::Zero(10000);
	if (level >= 4)
	{
		vector<vector<MatrixXd>> test_4 = vInterpolation["_4"];
		VectorXd V_4 = interTest.serial(TX, test_4[0], test_4[1], test_4[2], test_4[3]);
		vector<vector<MatrixXd>> test5 = vInterpolation["5"];
		VectorXd V5 = interTest.serial(TX, test5[0], test5[1], test5[2], test5[3]);
		U2 = V5 - V_4;
	}
	VectorXd U3 = VectorXd::Zero(10000);
	if (level >= 5)
	{
		vector<vector<MatrixXd>> test_5 = vInterpolation["_5"];
		VectorXd V_5 = interTest.serial(TX, test_5[0], test_5[1], test_5[2], test_5[3]);
		vector<vector<MatrixXd>> test6 = vInterpolation["6"];
		VectorXd V6 = interTest.serial(TX, test6[0], test6[1], test6[2], test6[3]);
		U3 = V6 - V_5;
	}
	VectorXd U4 = VectorXd::Zero(10000);
	if (level >= 6)
	{
		vector<vector<MatrixXd>> test_6 = vInterpolation["_6"];
		VectorXd V_6 = interTest.serial(TX, test_6[0], test_6[1], test_6[2], test_6[3]);
		vector<vector<MatrixXd>> test7 = vInterpolation["7"];
		VectorXd V7 = interTest.serial(TX, test7[0], test7[1], test7[2], test7[3]);
		U4 = V7 - V_6;
	}
	VectorXd U5 = VectorXd::Zero(10000);
	if (level >= 7)
	{
		vector<vector<MatrixXd>> test_7 = vInterpolation["_7"];
		VectorXd V_7 = interTest.serial(TX, test_7[0], test_7[1], test_7[2], test_7[3]);
		vector<vector<MatrixXd>> test8 = vInterpolation["8"];
		VectorXd V8 = interTest.serial(TX, test8[0], test8[1], test8[2], test8[3]);
		U5 = V8 - V_7;
	}
	VectorXd U6 = VectorXd::Zero(10000);
	if (level >= 8)
	{
		vector<vector<MatrixXd>> test_8 = vInterpolation["_8"];
		VectorXd V_8 = interTest.serial(TX, test_8[0], test_8[1], test_8[2], test_8[3]);
		vector<vector<MatrixXd>> test9 = vInterpolation["9"];
		VectorXd V9 = interTest.serial(TX, test9[0], test9[1], test9[2], test9[3]);
		U6 = V9 - V_8;
	}
	VectorXd U7 = VectorXd::Zero(10000);
	if (level >= 9)
	{
		vector<vector<MatrixXd>> test_9 = vInterpolation["_9"];
		VectorXd V_9 = interTest.serial(TX, test_9[0], test_9[1], test_9[2], test_9[3]);
		vector<vector<MatrixXd>> test10 = vInterpolation["10"];
		VectorXd V10 = interTest.serial(TX, test10[0], test10[1], test10[2], test10[3]);
		U7 = V10 - V_9;
	}
	VectorXd U8 = VectorXd::Zero(10000);
	if (level >= 10)
	{
		vector<vector<MatrixXd>> test_10 = vInterpolation["_10"];
		VectorXd V_10 = interTest.serial(TX, test_10[0], test_10[1], test_10[2], test_10[3]);
		vector<vector<MatrixXd>> test11 = vInterpolation["11"];
		VectorXd V11 = interTest.serial(TX, test11[0], test11[1], test11[2], test11[3]);
		U8 = V11 - V_10;
	}
	//Logger::WriteMessage("inter_test complete");
	Common::Logger("inter_test complete");

	//VectorXd U = interTest.GetResult("3") - interTest.GetResult("2");
	//VectorXd U1 = interTest.GetResult("4") - interTest.GetResult("_3");
	//VectorXd U2 = interTest.GetResult("5") - interTest.GetResult("_4");
	//VectorXd U3 = interTest.GetResult("6") - interTest.GetResult("_5");
	//VectorXd U4 = interTest.GetResult("7") - interTest.GetResult("_6");
	//VectorXd U5 = interTest.GetResult("8") - interTest.GetResult("_7");
	//VectorXd U6 = interTest.GetResult("9") - interTest.GetResult("_8");
	//VectorXd U7 = interTest.GetResult("10") - interTest.GetResult("_9");
	//VectorXd U8 = interTest.GetResult("11") - interTest.GetResult("_10");

	//[AP] = ECP(TX, r, sigma, T, E);
	VectorXd AP = ECP(TX, r, sigma, T, E);
	//Logger::WriteMessage("MuSik addition");
	Common::Logger("MuSIK addition");
	int m = U.rows();
	MatrixXd MuSIK = MatrixXd::Zero(m,9);
	MuSIK.col(0) = U;
	MuSIK.col(1) = U + U1;
	MuSIK.col(2) = U + U1 + U2;
	MuSIK.col(3) = U + U1 + U2 + U3;
	MuSIK.col(4) = U + U1 + U2 + U3 + U4;
	MuSIK.col(5) = U + U1 + U2 + U3 + U4 + U5;
	MuSIK.col(6) = U + U1 + U2 + U3 + U4 + U5 + U6;
	MuSIK.col(7) = U + U1 + U2 + U3 + U4 + U5 + U6 + U7;
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

	vector<VectorXd> result = { RMS, Max };
	return result;
}
double SparseGridCollocation::RootMeanSquare(VectorXd v)
{
	double rms = sqrt((v.array() * v.array()).sum() / v.size() );
	return rms;
}

