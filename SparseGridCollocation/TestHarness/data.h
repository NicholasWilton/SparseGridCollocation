#pragma once


#include "stdafx.h"


using namespace Eigen;
using namespace std;

class data {

public:
	static MatrixXd GetTX();

	static MatrixXd GetLamb2();

	static MatrixXd GetLamb3(int n);

	static MatrixXd GetLamb_3(int n);

	static MatrixXd GetLamb4(int n);

	static MatrixXd GetLamb_4(int n);

	static MatrixXd GetLamb5(int n);

	static MatrixXd GetLamb_5(int n);

	static MatrixXd GetLamb6(int n);

	static MatrixXd GetLamb_6(int n);

	static MatrixXd GetLamb7(int n);

	static MatrixXd GetLamb_7(int n);

	static MatrixXd GetLamb8(int n);

	static MatrixXd GetLamb_8(int n);

	static MatrixXd GetLamb9(int n);

	static MatrixXd GetLamb_9(int n);

	static MatrixXd GetLamb10(int n);

	static MatrixXd GetLamb_10(int n);

	static MatrixXd GetLamb11(int n);

	static vector<MatrixXd> GetC_10();

	static vector<MatrixXd> GetA_10();

	static void Load(string file, MatrixXd &matrix);
};