#include "stdafx.h"
#include "InterTest.h"
#include "RBF.h"
#include <thread>


using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using Eigen::Map;
using namespace Eigen;
using namespace std;

//#include "CppUnitTest.h"
//using namespace Microsoft::VisualStudio::CppUnitTestFramework;

Leicester::SparseGridCollocation::InterTest::InterTest()
{
}


Leicester::SparseGridCollocation::InterTest::~InterTest()
{
}

VectorXd Leicester::SparseGridCollocation::InterTest::GetResult(string id)
{
	return result->at(id);
}

map<string, VectorXd> Leicester::SparseGridCollocation::InterTest::GetResults()
{
	return *result;
}

void Leicester::SparseGridCollocation::InterTest::Execute(map<string, vector<vector<MatrixXd>> > vInterpolation, MatrixXd TX)
{
	vector<VectorXd> V;
	vector<thread> threads;
	

	int count = 0;
	for (auto v : vInterpolation)
	{
		threads.push_back(std::thread(&InterTest::parallel, this, v.first, TX, v.second[0], v.second[0], v.second[0], v.second[0]));
		//VectorXd v_ = inter_test( TX, v.second[0], v.second[0], v.second[0], v.second[0]);
		count++;
	}

	for (int i =0; i< threads.size(); i++)
	{
		threads.at(i).join();
	}

}

void Leicester::SparseGridCollocation::InterTest::parallel(string id, const MatrixXd &X, const vector<MatrixXd> &lamb,const vector<MatrixXd> &TX, const vector<MatrixXd> &C, const vector<MatrixXd> &A)
{
	// This is used to calculate values on final testing points
	//ch = length(TX);
	int ch = TX.size();

	//[N, ~] = size(X);
	int N = X.rows();
	//V = ones(N, ch);
	MatrixXd V = MatrixXd::Ones(N, ch);
	//for j = 1:ch

	for (int j = 0; j < ch; j++)
	{
		RBF r;
		//[D] = mq2d(X, TX{ j }, A{ j }, C{ j });
		vector<MatrixXd> D = r.Gaussian2D(X, TX[j], A[j], C[j]);
		//vector<MatrixXd> D = CudaRBF::Gaussian2D(X, TX[j], A[j], C[j]);
		//Common::saveArray(X, "Musikc_X.txt");
		//Common::saveArray(TX[j], "Musikc_TX.txt");
		//Common::saveArray(A[j], "Musikc_A.txt");
		//Common::saveArray(C[j], "Musikc_C.txt");
		//Common::saveArray(D[0], "Musikc_D.txt");
		//V(:, j) = D*lamb{ j };
		VectorXd v = D[0] * lamb[j];
		V.col(j) = v;
		//end
	}
	//output = sum(V, 2);
	VectorXd res = V.rowwise().sum().eval();
	auto r = map<string, VectorXd>::value_type(id, res);
	result->insert(r);
	
}

void Leicester::SparseGridCollocation::InterTest::parallelND(string id, const MatrixXd &X, const vector<MatrixXd> &lamb, const vector<MatrixXd> &TX, const vector<MatrixXd> &C, const vector<MatrixXd> &A)
{
	// This is used to calculate values on final testing points
	//ch = length(TX);
	int ch = TX.size();

	//[N, ~] = size(X);
	int N = X.rows();
	//V = ones(N, ch);
	MatrixXd V = MatrixXd::Ones(N, ch);
	//for j = 1:ch

	for (int j = 0; j < ch; j++)
	{
		RBF r;
		//[D] = mq2d(X, TX{ j }, A{ j }, C{ j });
		vector<MatrixXd> D = r.GaussianND(X, TX[j], A[j], C[j]);
		//vector<MatrixXd> D = r.mqd2(X, TX[j], A[j], C[j]);
		//Common::saveArray(X, "MusikcND_X.txt");
		//Common::saveArray(TX[j], "MusikcND_TX.txt");
		//Common::saveArray(A[j], "MusikcND_A.txt");
		//Common::saveArray(C[j], "MusikcND_C.txt");
		//Common::saveArray(D[0], "MusikcND_D.txt");
		//Common::saveArray(lamb[j], "MusikcND_lamb.txt");
		//V(:, j) = D*lamb{ j };
		VectorXd v = D[0] * lamb[j];
		V.col(j) = v;
		//end
	}
	//output = sum(V, 2);
	VectorXd res = V.rowwise().sum().eval();
	auto r = map<string, VectorXd>::value_type(id, res);
	result->insert(r);

}

VectorXd Leicester::SparseGridCollocation::InterTest::serial(MatrixXd X, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A)
{
	// This is used to calculate values on final testing points
	//ch = length(TX);
	int ch = TX.size();

	//[N, ~] = size(X);
	int N = X.rows();
	//V = ones(N, ch);
	MatrixXd V = MatrixXd::Ones(N, ch);
	//for j = 1:ch
	vector<MatrixXd> res;

	for (int j = 0; j < ch; j++)
	{
		RBF r;
		//[D] = mq2d(X, TX{ j }, A{ j }, C{ j });
		vector<MatrixXd> D = r.Gaussian2D(X, TX[j], A[j], C[j]);

		//V(:, j) = D*lamb{ j };
		VectorXd v = D[0] * lamb[j];
		V.col(j) = v;
		//end
	}
	//output = sum(V, 2);
	//Logger::WriteMessage(Common::printMatrix(V).c_str());
	VectorXd output = V.rowwise().sum();
	//Logger::WriteMessage(Common::printMatrix(output).c_str());
	return output;

}
