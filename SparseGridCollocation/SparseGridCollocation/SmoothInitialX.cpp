#include "stdafx.h"
#include "SmoothInitialX.h"
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

VectorXd Leicester::SparseGridCollocation::SmoothInitialX::x;

VectorXd Leicester::SparseGridCollocation::SmoothInitialX::X()
{
	if (x.size() == 0)
	{
		ifstream infile("SmoothInitialX.txt");
		vector<double> doubles;
		int count = 0;
		while (infile)
		{
			string s;
			if (!getline(infile, s)) break;

			doubles.push_back(stod(s));
			count++;
		}

		VectorXd X(doubles.size());
		vector<double>::iterator it;
		int i = 0;
		for (it = doubles.begin(); it < doubles.end(); it++, i++) {
			X(i) = doubles[i];
		}

		x = X;
	}
	return x;
}
