#include "stdafx.h"
#include "PDE.h"
#include "RBF.h"
#include "..\Common\Utility.h"
#include "MatrixXdm.h"
#include "kernel.h"
#include "Gaussian2d1.h"
//#include "GaussianNd1.h"
#include "Common.h"
#include <thread>


//#include "CppUnitTest.h"
//using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Leicester::Common;
using namespace Leicester::ThrustLib;

int Leicester::SparseGridCollocation::PDE::callCount;

Leicester::SparseGridCollocation::PDE::PDE()
{
}


Leicester::SparseGridCollocation::PDE::~PDE()
{
}

MatrixXd Leicester::SparseGridCollocation::PDE::BlackScholes(const MatrixXd &node, double r, double sigma,
	const vector<MatrixXd> &lambda2, const vector<MatrixXd> &TX2, const vector<MatrixXd> &C2, const vector<MatrixXd> &A2,
	const vector<MatrixXd> &lambda3, const vector<MatrixXd> &TX3, const vector<MatrixXd> &C3, const vector<MatrixXd> A3)
{
	int N = node.rows();
	int ch2 = TX2.size();
	MatrixXd U2 = MatrixXd::Ones(N, ch2);

	for (int j = 0; j < ch2; j++)
	{
		vector<MatrixXd> mqd = RBF::Gaussian2D(node, TX2[j], A2[j], C2[j]);
		MatrixXd a = mqd[1] * lambda2[j];
		MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * lambda2[j];
		MatrixXd c = r * mqd[2] * lambda2[j];
		MatrixXd d = r * mqd[0] * lambda2[j];
		U2.col(j) = a + b + c - d;
	}
	int ch3 = TX3.size();
	MatrixXd U3 = MatrixXd::Ones(N, ch3);
	for (int j = 0; j < ch3; j++)
	{
		vector<MatrixXd> mqd = RBF::Gaussian2D(node, TX3[j], A3[j], C3[j]);
		MatrixXd a = mqd[1] * lambda3[j];
		MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * lambda3[j];
		MatrixXd c = r * mqd[2] * lambda3[j];
		MatrixXd d = r * mqd[0] * lambda3[j];
		U3.col(j) = a + b + c - d;
	}
	MatrixXd s1 = U3.rowwise().sum();
	MatrixXd s2 = U2.rowwise().sum();
	MatrixXd output = s1 - s2;

	return output;
}

MatrixXd Leicester::SparseGridCollocation::PDE::BlackScholesC(const MatrixXd &node, double r, double sigma,
	const vector<MatrixXd> &lambda2, const vector<MatrixXd> &TX2, const vector<MatrixXd> &C2, const vector<MatrixXd> &A2,
	const vector<MatrixXd> &lambda3, const vector<MatrixXd> &TX3, const vector<MatrixXd> &C3, const vector<MatrixXd> A3)
{
	int N = node.rows();
	int ch2 = TX2.size();
	MatrixXd U2 = MatrixXd::Ones(N, ch2);
	unsigned int memory = node.rows() * node.cols() * sizeof(double) * 6;
	MemoryInfo mi = ThrustLib::Common::GetMemory();
	ThrustLib::Gaussian2d1* cudaGaussian = NULL;
	if (mi.free > memory)
		cudaGaussian = new Gaussian2d1(node);
	
	for (int j = 0; j < ch2; j++)
	{

		vector<MatrixXd> mqd;
		//if there is free memory then do RBF interpolation gpu else on the cpu
		if (cudaGaussian != NULL & (j % 2 == 0))
		{
			//wcout << "Sending matrix size=" << memory << " bytes to GPU" << endl;
			mqd = cudaGaussian->Gaussian2d(TX2[j], A2[j], C2[j]);
		}
		else
		{
			//wcout << "Sending matrix size=" << memory << " bytes to CPU" << endl;
			mqd = RBF::Gaussian2D(node, TX2[j], A2[j], C2[j]);;
		}

		MatrixXd a = mqd[1] * lambda2[j];
		MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * lambda2[j];
		MatrixXd c = r * mqd[2] * lambda2[j];
		MatrixXd d = r * mqd[0] * lambda2[j];
		U2.col(j) = a + b + c - d;
	}
	int ch3 = TX3.size();
	MatrixXd U3 = MatrixXd::Ones(N, ch3);
	for (int j = 0; j < ch3; j++)
	{
		vector<MatrixXd> mqd;
		//if there is free memory then do RBF interpolation gpu else on the cpu
		if (cudaGaussian != NULL & (j % 2 == 0))
		{
			//wcout << "Sending matrix size=" << memory << " bytes to GPU" << endl;
			mqd = cudaGaussian->Gaussian2d(TX3[j], A3[j], C3[j]);
		}
		else
		{
			//wcout << "Sending matrix size=" << memory << " bytes to CPU" << endl;
			mqd = RBF::Gaussian2D(node, TX3[j], A3[j], C3[j]);;
		}

		MatrixXd a = mqd[1] * lambda3[j];
		MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * lambda3[j];
		MatrixXd c = r * mqd[2] * lambda3[j];
		MatrixXd d = r * mqd[0] * lambda3[j];
		U3.col(j) = a + b + c - d;
	}
	MatrixXd s1 = U3.rowwise().sum();
	MatrixXd s2 = U2.rowwise().sum();
	MatrixXd output = s1 - s2;

	return output;
}

MatrixXd Leicester::SparseGridCollocation::PDE::BlackScholesNd(const MatrixXd &node, double r, double sigma, vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state)
{
	int N = node.rows();
	vector<MatrixXd> Us;
	//wcout << "keys=" << keys.size() << endl;
	for (auto key : keys)
	{
		//cout << "key:" << key << endl;
		vector<vector<MatrixXd>> item = state->at(key); //0-lambda, 1-TX, 2-C, 3-A, 4-U
		int ch2 = item[1].size();
		MatrixXd U = MatrixXd::Ones(N, ch2);
		
		stringstream ss;
		ss << "PDE." << key;
		//wcout << "ch2=" << ch2 << endl;
		for (int j = 0; j < ch2; j++)
		{
			//stringstream ssj;
			//ssj << ss.str() << "." << j;
			//stringstream ssk;
			//ssk << ssj.str() << ".node.txt";
			//Common::saveArray(node, ssk.str());
			//ssk.str(string());
			//ssk << ssj.str() << ".item1.txt";
			//Common::saveArray(item[1][j], ssk.str());
			//MatrixXd m = item[1][j];
			//wcout << Common::printMatrix(m) << endl;
			//ssk.str(string());
			//ssk << ssj.str() << ".item2.txt";
			//Common::saveArray(item[2][j], ssk.str());
			//ssk.str(string());
			//ssk << ssj.str() << ".item3.txt";
			//Common::saveArray(item[3][j], ssk.str());

			//Common::Utility::saveArray(node, "node_Nd.txt");
			//Common::Utility::saveArray(item[1][j], "CN_Nd.txt");
			//Common::Utility::saveArray(item[3][j], "A_Nd.txt");
			//Common::Utility::saveArray(item[2][j], "C_Nd.txt");
			PDE::callCount++;
			vector<MatrixXd> mqd = RBF::GaussianND(node, item[1][j], item[3][j], item[2][j]);
			//Common::Utility::saveArray(mqd[0], "mqd0.txt");
			//Common::Utility::saveArray(mqd[1], "mqd1.txt");
			//Common::Utility::saveArray(mqd[2], "mqd2.txt");
			//Common::Utility::saveArray(mqd[3], "mqd3.txt");

			//MatrixXd a = mqd[1] * item[0][j]; // lambda * dV/dt
			//MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * item[0][j]; // 1/2 sum-i sum-j sigma^2 rho-ij Si Sj d2V/dSi dSj
			//MatrixXd c = r * mqd[2] * item[0][j]; // sum-i (r - q-i) Si dV/dSi
			//MatrixXd d = r * mqd[0] * item[0][j]; //rV
			//U.col(j) = a + b + c - d;
			U.col(j) = (mqd[1] * item[0][j]) + ((pow(sigma, 2) / 2) * mqd[3] * item[0][j]) + (r * mqd[2] * item[0][j]) - (r * mqd[0] * item[0][j]);
			//ssk.str(string());
			//ssk << ssj.str() << ".ucolj.txt";
			//Common::saveArray(U.col(j), ssk.str());
		}
		Us.push_back(U);
	}

	int n = Us.size();
	MatrixXd output= MatrixXd::Zero(Us[0].rows(), 1);;
	for (int i = 0; i < Us.size(); i++)
	{
		int coeff = Leicester::Common::Utility::BinomialCoefficient(n-1, i);
		MatrixXd U = Us[i];
		VectorXd sum = U.rowwise().sum();
		//Common::saveArray(sum, "sum.txt");
		if (i % 2 == 0)
			output.col(0).array() += (coeff * sum).array();
		else
			output.col(0).array() -= (coeff * sum).array();
	}
	//Common::saveArray(output, "ouput.txt");

	return output;
}

MatrixXd Leicester::SparseGridCollocation::PDE::BlackScholesNdC(const MatrixXd &node, double r, double sigma, vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state)
{
	int N = node.rows();
	vector<MatrixXd> Us;
	//wcout << "keys=" << keys.size() << endl;
	for (auto key : keys)
	{
		vector<vector<MatrixXd>> item = state->at(key); //0-lambda, 1-TX, 2-C, 3-A, 4-U
		int ch2 = item[1].size();
		MatrixXd U = MatrixXd::Ones(N, ch2);
		//wcout << "ch2=" << ch2 << endl;
		stringstream ss;
		ss << "PDE." << key;
		unsigned int memory = node.rows() * node.cols() * sizeof(double) * 6;
		MemoryInfo mi = ThrustLib::Common::GetMemory();
		ThrustLib::GaussianNd1* cudaGaussian = NULL;
		if (mi.free > memory)
			cudaGaussian = new GaussianNd1(node);
		vector<thread> threads;
		map<int,MatrixXd> vd;
		map<int,MatrixXd> vdt;
		map<int,MatrixXd> vdx;
		map<int,MatrixXd> vdxx;
		for (int j = 0; j < ch2; j++)
		{
			vector<MatrixXd> mqd;
			if (cudaGaussian != NULL & (j % 2 == 0))
			{
				wcout << "Sending matrix size=" << memory << " bytes to GPU" << endl;
				//Common::Utility::saveArray(node, "node_Ndc.txt");
				//Common::Utility::saveArray(item[1][j], "CN_Ndc.txt");
				//Common::Utility::saveArray(item[3][j], "A_Ndc.txt");
				//Common::Utility::saveArray(item[2][j], "C_Ndc.txt");
				PDE::callCount++;
				//vector<MatrixXd> mq = RBF::GaussianND(node, item[1][j], item[3][j], item[2][j]);
				//MatrixXd d(node.rows(), item[1][j].rows());
				//vd[j] = d;
				//MatrixXd dt(node.rows(), item[1][j].rows());
				//vdt[j] =dt;
				//MatrixXd dx(node.rows(), item[1][j].rows());
				//vdx[j] = dx;
				//MatrixXd dxx(node.rows(), item[1][j].rows());
				//vdxx[j] = dxx;
				//threads.push_back(std::thread(&Leicester::SparseGridCollocation::PDE::GaussianNd, item[1][j], item[3][j], item[2][j], &d, &dt, &dx, &dxx, cudaGaussian));
				mqd = cudaGaussian->GaussianNd(item[1][j], item[3][j], item[2][j]);
				vd[j] = mqd[0];
				vdt[j] = mqd[1];
				vdx[j] = mqd[2];
				vdxx[j] = mqd[3];
				//Common::Utility::saveArray(mq[0], "cD1.txt");
				//Common::Utility::saveArray(mq[1], "cDt1.txt");
				//Common::Utility::saveArray(mq[2], "cDx1.txt");
				//Common::Utility::saveArray(mq[3], "cDxx1.txt");

				//bool f1 = Common::Utility::checkMatrix(mq[0], mqd[0], 0.001, false);
				////wcout << "Dt" << endl;
				//bool f2 = Common::Utility::checkMatrix(mq[1], mqd[1], 0.001, false);
				////wcout << "Dx" << endl;
				//bool f3 = Common::Utility::checkMatrix(mq[2], mqd[2], 0.001, false);
				////wcout << "Dxx" << endl;
				//bool f4 = Common::Utility::checkMatrix(mq[3], mqd[3], 0.001, false);
				//if (!f1 | !f2 | !f3 | !f4)
				//{
				//	Common::Utility::saveArray(node, "cTX1.txt");
				//	Common::Utility::saveArray(item[1][j], "cCN1.txt");
				//	Common::Utility::saveArray(item[3][j], "cA.txt");
				//	Common::Utility::saveArray(item[2][j], "cC.txt");
				//	Common::Utility::saveArray(mqd[0], "cD.txt");
				//	Common::Utility::saveArray(mqd[1], "cDt.txt");
				//	Common::Utility::saveArray(mqd[2], "cDx.txt");
				//	Common::Utility::saveArray(mqd[3], "cDxx.txt");
				//	Common::Utility::saveArray(mq[0], "cD1.txt");
				//	Common::Utility::saveArray(mq[1], "cDt1.txt");
				//	Common::Utility::saveArray(mq[2], "cDx1.txt");
				//	Common::Utility::saveArray(mq[3], "cDxx1.txt");
				//}
				//U.col(j) = (mqd[1] * item[0][j]) + ((pow(sigma, 2) / 2) * mqd[3] * item[0][j]) + (r * mqd[2] * item[0][j]) - (r * mqd[0] * item[0][j]);
				
			}
			else
			{
				wcout << "Sending matrix size=" << memory << " bytes to CPU" << endl;
				mqd = RBF::GaussianND(node, item[1][j], item[3][j], item[2][j]);
				vd[j] =mqd[0];
				vdt[j] = mqd[1];
				vdx[j] = mqd[2];
				vdxx[j] =mqd[3];
				//U.col(j) = (mqd[1] * item[0][j]) + ((pow(sigma, 2) / 2) * mqd[3] * item[0][j]) + (r * mqd[2] * item[0][j]) - (r * mqd[0] * item[0][j]);
			}
		}

		for (int i = 0; i < threads.size(); i++)
			threads.at(i).join();

		for (int j = 0; j < ch2; j++)
		{
			MatrixXd d = vd[j];
			MatrixXd dt = vdt[j];
			MatrixXd dx = vdx[j];
			MatrixXd dxx = vdxx[j];
			//if there was a cuda memcopy error, recover by re-running on the CPU:
			if ((d.rows() == 0 & d.cols() == 0) | (dt.rows() == 0 & dt.cols() == 0) | (dx.rows() == 0 & dx.cols() == 0) | (dxx.rows() == 0 & dxx.cols() == 0))
			{
				vector<MatrixXd> mqd = RBF::GaussianND(node, item[1][j], item[3][j], item[2][j]);
				d = mqd[j];
				dt = mqd[j];
				dx = mqd[j];
				dxx = mqd[j];
			}
			U.col(j) = (dt * item[0][j]) + ((pow(sigma, 2) / 2) * dxx * item[0][j]) + (r * dx * item[0][j]) - (r * d * item[0][j]);
		}
		Us.push_back(U);

		delete cudaGaussian;
	}

	int n = Us.size();
	MatrixXd output = MatrixXd::Zero(Us[0].rows(), 1);;
	for (int i = 0; i < Us.size(); i++)
	{
		int coeff = Leicester::Common::Utility::BinomialCoefficient(n - 1, i);
		MatrixXd U = Us[i];
		VectorXd sum = U.rowwise().sum();
		if (i % 2 == 0)
			output.col(0).array() += (coeff * sum).array();
		else
			output.col(0).array() -= (coeff * sum).array();
	}

	return output;
}

void Leicester::SparseGridCollocation::PDE::GaussianNd(const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C, 
	MatrixXd* d, MatrixXd* dt, MatrixXd* dx, MatrixXd* dxx, ThrustLib::GaussianNd1* cudaGaussian)
{
	try
	{
		vector<MatrixXd> result = cudaGaussian->GaussianNd(CN, A, C);
		d = new MatrixXd(result[0]);
		dt = new MatrixXd(result[1]);
		dx = new MatrixXd(result[2]);
		dxx = new MatrixXd(result[3]);
	}
	catch (...)
	{
		d = new MatrixXd(0,0);
		dt = new MatrixXd(0,0);
		dx = new MatrixXd(0,0);
		dxx = new MatrixXd(0,0);
	}
}