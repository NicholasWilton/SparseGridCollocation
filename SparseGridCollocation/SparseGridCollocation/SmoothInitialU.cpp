#include "stdafx.h"
#include "SmoothInitialU.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using namespace std;

VectorXd Leicester::SmoothInitialU::u;

VectorXd Leicester::SmoothInitialU::U()
{
	if (u.size() == 0)
	{
		ifstream infile("SmoothInitialU.txt");
		vector<double> doubles;
		int count = 0;
		while (infile)
		{
			string s;
			if (!getline(infile, s)) break;

			doubles.push_back(stod(s));
			count++;
		}

		VectorXd U(doubles.size());
		vector<double>::iterator it;
		int i = 0;
		for (it = doubles.begin(); it < doubles.end(); it++, i++) {
			U(i) = doubles[i];
		}

		u = U;
	}

	return u;
}
