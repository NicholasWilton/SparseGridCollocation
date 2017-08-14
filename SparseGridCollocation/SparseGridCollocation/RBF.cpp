#include "stdafx.h"
#include "RBF.h"
#include <math.h>
#include "Common.h"
#include <iomanip>

RBF::RBF()
{
}


RBF::~RBF()
{
}

VectorXd RBF::exp(const VectorXd &v)
{
	VectorXd result(v.rows());
	for (int i = 0; i < v.size(); i++)
	{
		result[i] = std::exp(v[i]);
	}
	return result;
}

vector<MatrixXd> RBF::mqd2(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C)
{
	vector<MatrixXd> result;
	int Num = CN.rows();
	int N = TP.rows();

	MatrixXd D(N, Num);
	D.fill(1.0);
	MatrixXd Dt(N, Num);
	Dt.fill(1.0);
	MatrixXd Dx(N, Num);
	Dx.fill(1.0);
	MatrixXd Dxx(N, Num);
	Dxx.fill(1.0);

	for (int j = 0; j < Num; j++)
	{
		VectorXd a1 = A(0, 0)*(TP.col(0).array() - CN(j, 0)); // a * (t - cn)
		VectorXd b1 = -(a1.array() * a1.array()) / (C(0, 0) *C(0, 0)); //( a * (t - cn)  )^2 / c^2
		VectorXd FAI1 = b1.array().exp(); //V(t) = exp[ ( a * (t - cn)  )^2 / c^2 ]
		//VectorXd FAI1 = RBF::exp(b1);

		VectorXd a2 = A(0, 1)*(TP.col(1).array() - CN(j, 1));
		VectorXd b2 = -(a2.array() * a2.array()) / (C(0, 1) *C(0, 1));

		VectorXd FAI2 = b2.array().exp(); //V(x) = exp[(a * (x - cn)) ^ 2 / c ^ 2]
		//VectorXd FAI2 = RBF::exp(b2);
		D.col(j) = FAI1.array() * FAI2.array(); //V(x,t) = V(x)*V(t) = exp[(a * (x - cn)) ^ 2 / c ^ 2] * exp[ ( a * (t - cn)  )^2 / c^2 ]

		VectorXd a3 = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0)) * (TP.col(0).array() - CN(j, 0)); // -2 *[ (a/c)^2 * (t - cn) ]
		VectorXd b3 = a3.array() * FAI1.array(); // -2 *[ (a/c)^2 * (t - cn) ] * exp[ ( a * (t - cn)  )^2 / c^2 ]
		VectorXd c3 = b3.array() * FAI2.array();//dV/dt = -2 *[ (a/c)^2 * (t - cn) ] * exp[ ( a * (t - cn)  )^2 / c^2 ] * exp[(a * (x - cn)) ^ 2 / c ^ 2]
		Dt.col(j) = c3;

		VectorXd a4 = -2 * (A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)) * (TP.col(1).array() - CN(j, 1)); // -2 *[ (a/c)^2 * (x - cn) ]
		VectorXd b4 = TP.col(1).array() * a4.array() * FAI1.array(); // x * ( -2 *[ (a/c)^2 * (x - cn) ] ) exp[ ( a * (t - cn)  )^2 / c^2 ]
		VectorXd c4 = b4.array() * FAI2.array(); //dV/dx = x * ( -2 *[ (a/c)^2 * (x - cn) ] ) exp[ ( a * (t - cn)  )^2 / c^2 ] exp[(a * (x - cn)) ^ 2 / c ^ 2]
		Dx.col(j) = c4;

		double sA = A(0, 1) * A(0, 1);
		double qA = A(0, 1) * A(0, 1) * A(0, 1) * A(0, 1); //a^4
		double sC = C(0, 1) * C(0, 1);
		double qC = C(0, 1) * C(0, 1) * C(0, 1) * C(0, 1); //c^4
		VectorXd dTpCn = TP.col(1).array() - CN(j, 1); // x - c

		VectorXd a5 = 4 * qA * (dTpCn.array() * dTpCn.array() / qC); // 4 * a^4 * (x - c)^2 / c^4
		VectorXd b5 = -2 * sA / sC + a5.array();//  -2 * a^2/c^2 + [4 * a^4 * (x - c)^2 / c^4]
		VectorXd c5 = b5.array()  * FAI2.array() * FAI1.array(); // ( -2 * a ^ 2 / c ^ 2 + [4 * a ^ 4 * (x - c) ^ 2 / c ^ 4] ) * exp[(a * (x - cn)) ^ 2 / c ^ 2] * exp[ ( a * (t - cn)  )^2 / c^2 ]
		VectorXd d5 = (TP.col(1).array() * TP.col(1).array()).array() * c5.array(); // d^2V/dx^2 = x^2 * ( -2 * a ^ 2 / c ^ 2 + [4 * a ^ 4 * (x - c) ^ 2 / c ^ 4] ) * exp[(a * (x - cn)) ^ 2 / c ^ 2] * exp[ ( a * (t - cn)  )^2 / c^2 ]
		Dxx.col(j) = d5;
	}

	result.push_back(D);
	result.push_back(Dt);
	result.push_back(Dx);
	result.push_back(Dxx);
	return result;
}

vector<MatrixXd> RBF::mqNd(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C)
{
	vector<MatrixXd> result;// V, Vt, Vx Vxy
	int Num = CN.rows();
	int N = TP.rows();
	int dimensions = TP.cols();

	MatrixXd D(N, Num);
	D.fill(1.0);

	vector<MatrixXd> Derivatives;
	Derivatives.push_back(D);
	for (int d = 0; d < 3; d++)
	{
		MatrixXd Dx(N, Num);
		Dx.fill(1.0);
		Derivatives.push_back(Dx);
	}

	
	for (int j = 0; j < Num; j++)
	{
		vector<VectorXd> FAIn;
		for (int d = 0; d < dimensions; d++)
		{
			//V
			VectorXd a1 = A(0, d)*(TP.col(d).array() - CN(j, d));
			VectorXd b1 = -(a1.array() * a1.array()) / (C(0, d) *C(0, d));
			VectorXd FAI = b1.array().exp();
			Derivatives[0].col(j).array() *= FAI.array();
			FAIn.push_back(FAI);
		}

		//Vt
		VectorXd vt = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0)) * (TP.col(0).array() - CN(j, 0));
		Derivatives[1].col(j) = vt;
		//Common::saveArray(vt, "vt.txt");

		//sum-i (r - q-i) * S-i * dV/dS-i
		MatrixXd dS(TP.rows(), dimensions -1);
		for (int d = 1; d < dimensions; d++)
		{
			VectorXd a4 = -2 * (A(0, d) / C(0, d)) * (A(0, d) / C(0, d)) * (TP.col(d).array() - CN(j, d)); // -2 *[ (a/c)^2 * (x - cn) ]
			dS.col(d-1) = a4.array() * TP.col(d).array();
		}
		VectorXd sum = dS.rowwise().sum();
		Derivatives[2].col(j) = sum;

		// d^2V/dS-i dS-j =  4 * [ (a/c)^4 * (S-i * S-j + cn^2 - S-j*cn - Sj* cn) ] * exp[(a * (S-i - cn)) ^ 2 / c ^ 2] * exp[ ( a * (S-j - cn)  )^2 / c^2 ]
		// 1/2 * sum-i sum-j [sigma-i * sigma-j *rho-ij *S-i *S-j * d^2V/dS-i dS-j]
		VectorXd sumij = VectorXd::Zero(TP.rows());
		for (int d = 1; d < dimensions; d++) 
		{
			VectorXd sumi = VectorXd::Zero(TP.rows());
			for (int i = 1; i < TP.cols(); i++)
			{
				VectorXd vxy;
				double sA = A(0, d) * A(0, d);
				double qA = sA * sA;
				VectorXd diff = TP.col(d).array() - CN(j, d);
				double sC = C(0, d) *C(0, d) ;
				double qC = sC * sC;
				VectorXd dTpCn = TP.col(d).array() - CN(j, i); // x - c
				//double dAC = A(0, d) / C(0, d);
				//TODO: do we need to transpose TP.row() to get a column vec?
				//vxy = 4 * (qA / qC) * (TP.col(d).array() * TP(i,d) + (CN(j, d) * CN(j, d)) - (TP.col(d).array() * CN(j, d)) - (TP(i, d) * CN(j, d)));
				VectorXd a5 = 4 * qA * (dTpCn.array() * dTpCn.array() / qC); // 4 * a^4 * (x - c)^2 / c^4
				vxy = -2 * sA / sC + a5.array();//  -2 * a^2/c^2 + [4 * a^4 * (x - c)^2 / c^4]
				sumi.array() = sumi.array() + TP.col(d).array() * TP.col(i).array() * vxy.array();
			}
			sumij.array() = sumij.array() + sumi.array();
			
		}
		Derivatives[3].col(j) = sumij;

		for (int d = 1; d < Derivatives.size(); d++)
			Derivatives[d].col(j).array() *= Derivatives[0].col(j).array();
	
	}
	//int count = 1;
	//for (auto d : Derivatives)
	//{
	//	stringstream ss;
	//	ss << "d" << count << ".txt";
	//	Common::saveArray(d, ss.str());
	//	count++;
	//}

	//V, dV/dt, dV/ds-i, d^2V/dS-i*dS-j
	return Derivatives;
}

vector<MatrixXd> RBF::mqd1(const MatrixXd &x, double xc, const double c)
{
	//wcout << setprecision(25) << c << endl;
	vector<MatrixXd> result;

	MatrixXd r = x.array() - xc;
	
	MatrixXd phi = ((r.array()* r.array()) + (c * c)).sqrt();

	MatrixXd phi1 = x.array() * r.array() / phi.array();
	
	//wcout << Common::printMatrix(x) << endl;
	//wcout << Common::printMatrix(phi) << endl;
	//wcout << setprecision(25) << c << endl;

	MatrixXd phi2 = x.array() * x.array() * c * c / (phi.array() * phi.array() * phi.array());

	MatrixXd phi3 = -3 * c * c * r.array() / (phi.array() * phi.array() * phi.array() * phi.array() * phi.array());
	
	//wcout << Common::printMatrix(phi) << endl;
	result.push_back(phi);
	//wcout << Common::printMatrix(phi1) << endl;
	result.push_back(phi1);
	//wcout << Common::printMatrix(phi2) << endl;
	result.push_back(phi2);
	//wcout << Common::printMatrix(phi3) << endl;
	result.push_back(phi3);

	return result;
}