// SparseGridGollocation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialX.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using Eigen::Map;
using namespace std;

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


double prod(vector<double> x)
{
	double prod = 1.0;

	for (unsigned int i = 0; i < x.size(); i++)
		prod *= x[i];
	return prod;
}

template <typename T = double>
vector<T> linspace(T a, T b, size_t N) 
{
	T h = (b - a) / static_cast<T>(N - 1);
	vector<T> xs(N);
	typename vector<T>::iterator x;
	T val;
	for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
		*x = val;
	return xs;
}



vector<MatrixXd> mqd2(MatrixXd TP, MatrixXd CN, Matrix<double, 1, 2> A, MatrixXd C)
{
	vector<MatrixXd> *result = new vector<MatrixXd>();
	//[Num, ~] = size(CN);
	int Num = CN.rows();
	//[N, ~] = size(TP);
	int N = TP.rows();
	//[D, Dt, Dx, Dxx] = deal(ones(N, Num));
	
	MatrixXd D(N, Num);
	D.fill(1.0);
	MatrixXd Dt(N, Num);
	Dt.fill(1.0);
	MatrixXd Dx(N, Num);
	Dx.fill(1.0);
	MatrixXd Dxx(N, Num);
	Dxx.fill(1.0);

	//for j = 1:Num
	for (int j = 0; j < Num; j++)
	{
		cout << "TP.col" << endl;
		cout << TP.col(0) << endl;
		//	FAI1 = exp(-(A(1, 1)*(TP(:, 1) - CN(j, 1))). ^ 2 / C(1, 1) ^ 2);
		VectorXd a1 = A(0, 0)*(TP.col(0).array() - CN(j, 0));
		cout << "a1" << endl;
		cout << a1 << endl;
		VectorXd b1 = -(a1.array() * a1.array()) / (C(0, 0) *C(0, 0));
		cout << "b1" << endl;
		cout << b1 << endl;
		VectorXd FAI1 = b1.array().exp();
		
		//FAI2 = exp(-(A(1, 2)*(TP(:, 2) - CN(j, 2))). ^ 2 / C(1, 2) ^ 2);
		VectorXd a2 = A(0, 1)*(TP.col(1).array() - CN(j, 0));
		cout << "a2" << endl;
		cout << a2 << endl;
		VectorXd b2 = -(a2.array() * a2.array()) / (C(0, 0) *C(0, 0));
		cout << "b2" << endl;
		cout << b2 << endl;
		VectorXd FAI2 = b2.array().exp();
		//D(:, j) = FAI1.*FAI2;
		D.col(j) = FAI1.array() * FAI2.array();
		//TODO: this is basically how Matlab handles overloading:
		//if nargout > 1
		cout << "FAI1" << endl;
		cout << FAI1 << endl;
		cout << "FAI2" << endl;
		cout << FAI2 << endl;
		cout << "D" << endl;
		cout << D << endl;

		//	Dt(:, j) = -2 * (A(1, 1) / C(1, 1)) ^ 2 * (TP(:, 1) - CN(j, 1)).*FAI1.*FAI2;
		VectorXd a3 = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0)) * (TP.col(0).array() - CN(j, 0));
		VectorXd b3 = a3.array() * FAI1.array();
		VectorXd c3 = b3.array() * FAI2.array();
		Dt.col(j) = c3;
		//Dx(:, j) = TP(:, 2).*(-2 * (A(1, 2) / C(1, 2)) ^ 2 * (TP(:, 2) - CN(j, 2)).*FAI1.*FAI2);
		VectorXd a4 = -2 * (A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)) * (TP.col(1).array() - CN(j, 1));
		VectorXd b4 = TP.col(1).array() * a4.array() * FAI1.array();
		VectorXd c4 = b4.array() * FAI2.array();
		Dx.col(j) = c3;
		//Dxx(:, j) = TP(:, 2).^2.*((-2 * A(1, 2) ^ 2 / C(1, 2) ^ 2 + 4 * A(1, 2) ^ 4 * (TP(:, 2) - CN(j, 2)).^2. / C(1, 2) ^ 4).*FAI2.*FAI1);
		double a5 = -2 * (A(0, 1) * A(0, 1) / C(0, 1) * C(0, 1)) + 4 * A(0, 1) *A(0, 1) * A(0, 1)* A(0, 1) ;
		VectorXd b5 = TP.col(1).array() * a5 * (TP.col(1).array() - CN(j, 1)).array()* (TP.col(1).array() - CN(j, 1)).array();
		VectorXd c5 = b5.array() * FAI1.array() * FAI2.array();
		Dxx.col(j) = c5;
	}
	result->push_back(D);
	result->push_back(Dt);
	result->push_back(Dx);
	result->push_back(Dxx);
	return *result;
}

void shapelambda2D(MatrixXd lamb, MatrixXd *TX, MatrixXd C, MatrixXd A, double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, Matrix<double, 1, 2> N )
{
	//Num=prod(N);
	double num= N.prod();

	//t = linspace(0, tsec, N(1, 1));
	VectorXd t = VectorXd::LinSpaced(N(0, 0), 0, tsec);
	//x = linspace(inx1, inx2, N(1, 2));
	VectorXd x = VectorXd::LinSpaced(N(0, 1), inx1, inx2);

	//h1 = coef*tsec;
	double h1 = coef*tsec;
	//h2 = coef*(inx2 - inx1);
	double h2 = coef*(inx2 - inx1);
	
	//C = [h1, h2];
	//possible truncation here:
	 C = MatrixXd((int)h1, (int)h2);
	
	//A = N - 1;
	A = N.array() - 1;

	//[XXX, YYY] = meshgrid(t, x);
	/*
	XXX = RowVectorXd::LinSpaced(1, 3, 3).replicate(5, 1);
	YYY = VectorXd::LinSpaced(10, 14, 5).replicate(1, 3);
	*/

	MatrixXd XXX = t.replicate(1, x.rows());
	MatrixXd YYY = x.replicate(1, t.rows());

	XXX.transposeInPlace();
	//YYY.transposeInPlace();

	
	VectorXd xxx(Map<VectorXd>(XXX.data(), XXX.cols()*XXX.rows()));
	VectorXd yyy(Map<VectorXd>(YYY.data(), YYY.cols()*YYY.rows()));

	//TX = [XXX(:) YYY(:)];
	MatrixXd TX1(XXX.rows() * XXX.cols(), 2);// = new MatrixXd(15, 2);
	TX1 << xxx, yyy;
	
	//U=zeros(Num,1);
	VectorXd U = MatrixXd::Zero(num, 1);

	//int Num = TX1.rows();
	//int a = TX1.rows();
	//MatrixXd Dxx(a, Num);
	//Dxx.fill(1.0);

	//[ FAI, FAI_t, FAI_x, FAI_xx ] = mq2d( TX, TX, A, C );
	//MatrixXd* FAI = new MatrixXd(0, 0), *FAI_t = new MatrixXd(0, 0), *FAI_x = new MatrixXd(0, 0), *FAI_xx = new MatrixXd(0, 0);
	vector<MatrixXd> result = mqd2(TX1, TX1, A, C);
	MatrixXd FAI = result[0];
	MatrixXd FAI_t = result[1];
	MatrixXd FAI_x = result[2];
	MatrixXd FAI_xx = result[3];

	cout << "FAI" << endl;
	cout << FAI << endl;
	cout << "FAI_x" << endl;
	cout << FAI_x << endl;
	cout << "FAI_xx" << endl;
	cout << FAI_xx << endl;
	cout << "FAI_t" << endl;
	cout << FAI_t << endl;


	//P = FAI_t + sigma ^ 2 * FAI_xx / 2 + r*FAI_x - r*FAI;
	MatrixXd P = FAI_t.array() + (sigma * sigma * FAI_xx.array() / 2).array() + r*FAI_x.array() - r*FAI.array();

	//for i=1:Num
	for (int i = 0; i < num; i++)
	{
		//if TX(i,2) == inx1 || TX(i,2) == inx2       
		if (TX1(i, 1) == inx1 || TX1(i, 1) == inx2)
		{
			//P(i,:) = FAI(i,:);      
			P.row(i) = FAI.row(i);
			//U(i)=max( 0, TX(i,2) - E*exp( -r * (T-TX(i,1)) ) );
			double max = TX1(i, 1) - E*exp(-r * (T - TX1(i, 0)));
			U(i) = 0;
			if (max > 0)
				U(i) = max;

		}

		//if TX(i, 1) == tsec
		if (TX1(i, 0) == tsec)
		{
			//P(i, :) = FAI(i, :);
			P.row(i) = FAI.row(i);

			//U(i) = PPP(TX(i, :));
			//TX(i, :) is never used in PPP():
			//U(i) = PPP::Calculate(TX.row(i));
			U(i) = PPP::Calculate();
		}
	}
	TX = &TX1;
	//[F, J] = lu(P);
	MatrixXd F = P.lu().matrixLU().triangularView<UpLoType::Upper>();
	MatrixXd J = P.lu().matrixLU().triangularView<UpLoType::StrictlyLower>();
	cout << "P" << endl;
	cout << P << endl;
	cout << "F" << endl;
	cout << F << endl;
	cout << "J" << endl;
	cout << J << endl;
	//Jlamda = F\U;
	MatrixXd Jlamda = F.lu().solve(U);
	//lamb = J\Jlamda;
	lamb = J.lu().solve(Jlamda);

	cout << "lamb" << endl;
	cout << lamb << endl;

	return;
}

void interplant()
{
}

void test()
{
	vector <vector <string> > data;
	ifstream infile("SmoothInitialX.txt");

	while (infile)
	{
		string s;
		if (!getline(infile, s)) break;

		istringstream ss(s);
		vector <string> record;

		while (ss)
		{
			string s;
			if (!getline(ss, s, ',')) break;
			record.push_back(s);
		}

		data.push_back(record);
	}
	if (!infile.eof())
	{
		cerr << "Fooey!\n";
	}
}

int main()
{
	int ch = 10000;
	double inx1 = 0;
	double inx2 = 300;
	VectorXd x = VectorXd::LinSpaced(ch, inx1, inx2);
	VectorXd t(ch);
	t.fill(0);

	MatrixXd lamb(2, 1);
	MatrixXd TX(ch,2);
	TX << t, x;
	MatrixXd *TX3 = new MatrixXd(0,0);
	MatrixXd C(2, 1);
	MatrixXd A(2, 1);
	Matrix<double, 2, 2> N;
	N << 3, 5, 5, 3;
	for (int i =0; i < 2; i++)
		shapelambda2D(lamb, TX3, C, A, 2, 0.8650, 0.03, 0.15, 1, 100, 0, 300, N.row(i));

	return 0;
}