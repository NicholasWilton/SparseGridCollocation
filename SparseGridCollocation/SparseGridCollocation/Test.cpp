#include "stdafx.h"
#include "Test.h"
#include "RBF.h"
#include "Math.h"

#include "Common.h"
#include "CppUnitTest.h"
#include "Double.h"
#include "MatrixXdm.h"
//#include <MatrixFunctions>

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip> 
#include <fstream>

//using namespace Microsoft::VisualStudio::CppUnitTestFramework;

map<string, double> Test::mock;

Test::Test()
{
}


Test::~Test()
{
}

MatrixXd Test::MatrixExp(MatrixXd m)
{
	double e = 2.718281828459046;
	//double e = 2.71828182845904523536028747135266249775724709369995;
	MatrixXd res(m.rows(), m.cols());
	for (int i = 0; i < m.rows(); i++)
	{
		for (int j = 0; j < m.cols(); j++)
		{

			double exp = std::pow(e , m(i, j));

			res(i, j) = MatLabRounding(exp);
			//res(i, j) = exp;
		}
	}
	return res;
}

MatrixXd Test::MatLabRounding(MatrixXd m)
{
	MatrixXd res(m.rows(), m.cols());
	for (int i = 0; i < m.rows(); i++)
	{
		for (int j = 0; j < m.cols(); j++)
		{
			res(i, j) = MatLabRounding(m(i,j));
		}
	}
	return res;
}

double Test::MatLabRounding(double d)
{
	return round(d * 1000000000000000) / 1000000000000000;
}

map<string, double> Test::LoadMock()
{
	int n = 100000;
	//Data xx;
	char *key = new char[n];
	int *thread = new int[n];
	int *num = new int[n];
	int *sub = new int[n];
	double *value = new double[n];

	fstream kFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_key.dat", ios::in | ios::out | ios::binary);
	kFile.seekg(0);
	kFile.read((char*)key, sizeof(char) * n);

	fstream tFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_thread.dat", ios::in | ios::out | ios::binary);
	tFile.seekg(0);
	tFile.read((char*)thread, sizeof(int) * n);

	fstream nFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_num.dat", ios::in | ios::out | ios::binary);
	nFile.seekg(0);
	nFile.read((char*)num, sizeof(int) * n);

	fstream sFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_sub.dat", ios::in | ios::out | ios::binary);
	sFile.seekg(0);
	sFile.read((char*)sub, sizeof(int) * n);

	fstream vFile("C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\inner_test_value.dat", ios::in | ios::out | ios::binary);
	vFile.seekg(0);
	vFile.read((char*)value, sizeof(double) * n);

	map<string, double> result;

	for (int i = 0; i < n; i++)
	{
		stringstream ss;
		ss << key[i] << '_' << thread[i] << '_' << num[i] << '_' << sub[i];
		string s = ss.str();
		result[s] = value[i];
	}
	return result;
}

double Test::innerMock(string key, int thread, int num, int sub)
{
	if (Test::mock.size() == 0)
		Test::mock = LoadMock();
	stringstream ss;
	string k;
	if (key == "4")
		k = "\x1";
	if (key == "_3")
		k = "\x2";
	if (key == "5")
		k = "\x3";
	if (key == "_4")
		k = "\x4";
	if (key == "6")
		k = "\x5";
	if (key == "_5")
		k = "\x6";
	if (key == "7")
		k = "\x7";
	if (key == "_6")
		k = "\x8";
	if (key == "8")
		k = "\x9";
	if (key == "_7")
		k = "\x10";
	if (key == "9")
		k = "\x11";
	if (key == "_8")
		k = "\x12";
	if (key == "10")
		k = "\x13";
	if (key == "_9")
		k = "\x14";
	if (key == "11")
		k = "\x15";
	if (key == "_10")
		k = "\x16";
	ss << k << '_' << thread << '_' << num << '_' << sub;
	return Test::mock[ss.str()];
	//return 0;
}

double Test::inner(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A)
{
	// This is used in the PDE system re - construct for initial and boundary conditions
	int ch = TX.size();
	vector<double> V;
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
		MatrixXd a = -square1 / (C[j](0, 0) * C[j](0, 0));
		//Logger::WriteMessage(Common::printMatrix(a).c_str());
		MatrixXd FAI1 = MatrixExp(a);
		double e = exp(1);
		//Logger::WriteMessage(Common::printMatrixA(FAI1).c_str());
		//wcout << Common::printMatrix(FAI1) << endl;

		//FAI2 = exp(-(A{ j }(1, 2)*(x - TX{ j }(:, 2))). ^ 2 / C{ j }(1, 2) ^ 2);
		MatrixXd square2 = (A[j](0, 1) * (t - (TX[j].col(1).array())).array()) * (A[j](0, 1) * (t - (TX[j].col(1).array())).array());
		MatrixXd b = -square2 / (C[j](0, 1) * C[j](0, 1));
		MatrixXd FAI2 = MatrixExp(b);
		//Logger::WriteMessage(Common::printMatrixA(FAI2).c_str());
		//wcout << Common::printMatrix(FAI2) << endl;

		//D = FAI1.*FAI2;
		VectorXd D = FAI1.cwiseProduct(FAI2).eval();
		//Logger::WriteMessage(Common::printMatrixA(D).c_str());
		//Logger::WriteMessage(Common::printMatrixA(lamb[j]).c_str());
		//wcout << Common::printMatrix(D) << endl;
		//wcout << Common::printMatrix(lamb[j]) << endl;
		//V(j) = D'*lamb{j};
		MatrixXd d = MatLabRounding(D.transpose());
		//Logger::WriteMessage(Common::printMatrix(d).c_str());
		MatrixXd res = d * lamb[j];
		//Logger::WriteMessage(Common::printMatrixA(res).c_str());
		wcout << Common::printMatrix(lamb[j]).c_str() << endl;
		wcout << Common::printMatrixHexW(lamb[j]).c_str() << endl;
		cout << Common::printMatrixA(d).c_str() << endl;
		wcout << Common::printMatrixHexW(d).c_str() << endl;
		V.push_back(res(0,0));
		cout << Common::printMatrixA(res).c_str() << endl;
		wcout << Common::printMatrixHexA(res).c_str() << endl;
		MatrixXdM d1 (d);

		MatrixXdM l1(lamb[j]);
		//Logger::WriteMessage(Common::printMatrixA(l1.value()).c_str());
		//Logger::WriteMessage(Common::printMatrix(d1.value()).c_str());
		wcout << Common::printMatrixHexA(l1.value()).c_str() << endl;
		wcout << Common::printMatrixHexA(d1.value()).c_str() << endl;

		MatrixXd res1 = (d1 * l1).value();
		//Logger::WriteMessage(Common::printMatrixA(res1).c_str());
		cout << Common::printMatrixA(res1).c_str() << endl;
		wcout << Common::printMatrixHexA(res1).c_str() << endl;

		//   .....................
		//end
	}
	
	//output = sum(V);
	double output = 0;
	int i = 0;
	for (vector<double>::iterator it = V.begin(); it < V.end(); it++, i++)
	{
		//Logger::WriteMessage(Common::printMatrix(V[i]).c_str());
		output += V[i];// .sum();

	}

	return output;
}

double Test::innerY(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A)
{

	int ch = TX.size();
	vector<double> V;
	
	for (int j = 0; j < ch; j++)
	{
		
		MatrixXd square1 = (A[j](0, 0) * (t - (TX[j].col(0).array())).array()) * (A[j](0, 0) * (t - (TX[j].col(0).array())).array());
		//Logger::WriteMessage(Common::printMatrix(square1).c_str());
		MatrixXd a = -square1 / (C[j](0, 0) * C[j](0, 0));
		//Logger::WriteMessage(Common::printMatrix(a).c_str());
		MatrixXd FAI1 = a.array().exp();
		//Logger::WriteMessage(Common::printMatrix(FAI1).c_str());
		double e = exp(1);
		
		MatrixXd square2 = (A[j](0, 1) * (t - (TX[j].col(1).array())).array()) * (A[j](0, 1) * (t - (TX[j].col(1).array())).array());
		//Logger::WriteMessage(Common::printMatrix(square2).c_str());
		MatrixXd b = -square2 / (C[j](0, 1) * C[j](0, 1));
		//Logger::WriteMessage(Common::printMatrix(b).c_str());
		MatrixXd FAI2 = b.array().exp();
		//Logger::WriteMessage(Common::printMatrix(FAI2).c_str());
		

		//D = FAI1.*FAI2;
		VectorXd D = FAI1.cwiseProduct(FAI2).eval();
		//Logger::WriteMessage(Common::printMatrix(D).c_str());
		
		MatrixXd d = D.transpose();
		//Logger::WriteMessage(Common::printMatrix(d).c_str());
		
		//Logger::WriteMessage(Common::printMatrix(lamb[j]).c_str());
		MatrixXd res = d * lamb[j];
		//Logger::WriteMessage(Common::printMatrix(res).c_str());
		
		V.push_back(res(0, 0));
		

		//   .....................
		//end
	}

	//output = sum(V);
	double output = 0;
	int i = 0;
	for (vector<double>::iterator it = V.begin(); it < V.end(); it++, i++)
	{
		//Logger::WriteMessage(Common::printMatrix(V[i]).c_str());
		output += V[i];// .sum();

	}

	return output;
}

double Test::innerX(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A)
{
	// This is used in the PDE system re - construct for initial and boundary conditions
	int ch = TX.size();
	vector<double> V;
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
		MatrixXd a = -square1 / (C[j](0, 0) * C[j](0, 0));
		
		MatrixXd FAI1 = MatrixExp(a);
		double e = exp(1);
		
		//FAI2 = exp(-(A{ j }(1, 2)*(x - TX{ j }(:, 2))). ^ 2 / C{ j }(1, 2) ^ 2);
		MatrixXd square2 = (A[j](0, 1) * (t - (TX[j].col(1).array())).array()) * (A[j](0, 1) * (t - (TX[j].col(1).array())).array());
		MatrixXd b = -square2 / (C[j](0, 1) * C[j](0, 1));
		MatrixXd FAI2 = MatrixExp(b);
		
		
		VectorXd D = FAI1.cwiseProduct(FAI2).eval();
		//V(j) = D'*lamb{j};
		MatrixXd d = MatLabRounding(D.transpose());
		
		/*MatrixXd res = d * lamb[j];
		V.push_back(res(0, 0));*/

		
		MatrixXdM d1(d);
		MatrixXdM l1(lamb[j]);
		MatrixXd res1 = (d1 * l1).value();
		V.push_back(res1(0, 0));
		//   .....................
		//end
	}

	//output = sum(V);
	double output = 0;
	int i = 0;
	for (vector<double>::iterator it = V.begin(); it < V.end(); it++, i++)
	{
		//Logger::WriteMessage(Common::printMatrix(V[i]).c_str());
		output += V[i];// .sum();

	}

	return output;
}

VectorXd Test::inter(MatrixXd X, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A)
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
		//[D] = mq2d(X, TX{ j }, A{ j }, C{ j });
		vector<MatrixXd> D = RBF::mqd2(X, TX[j], A[j], C[j]);

		//V(:, j) = D*lamb{ j };
		VectorXd v = D[0] * lamb[j];
		V.col(j) = v;
		//end
	}
	//output = sum(V, 2);
	VectorXd output = V.colwise().sum();
	return output;
}