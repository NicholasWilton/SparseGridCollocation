#pragma once
#include "C:\Users\User\Source\Repos\SparseGridCollocation\SparseGridCollocation\include\eigen-eigen-67e894c6cd8f\Eigen\StdVector"

#define API _declspec(dllexport)

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace Eigen;


class API MethodOfLines
{
public:
	static int MoLiteration(double Tend, double Tdone, double dt, double *G, int GRows, int GCols, double *lamb, int lambRows, int lambCols, double inx2, double r, double K, MatrixXd A1, MatrixXd Aend, MatrixXd H);
};