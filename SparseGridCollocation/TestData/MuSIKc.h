#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>

using namespace Eigen;
using namespace std;

class MuSIKc
{
public:
	MuSIKc();
	~MuSIKc();


	

	static void Load(string file, MatrixXd &matrix)
	{
		ifstream infile(file);
		vector<double> doubles;
		int count = 0;
		while (infile)
		{
			string s;
			if (!getline(infile, s)) break;

			stringstream ss(s);
			while (ss.good())
			{
				string substr;
				getline(ss, substr, ',');
				doubles.push_back(stod(substr));
			}

			count++;
		}

		int rows = matrix.rows();
		int cols = matrix.cols();
		vector<double>::iterator it;
		int i = 0;
		int rcount = 0;
		int ccount = 0;

		for (it = doubles.begin(); it < doubles.end(); it++, i++) {
			if (ccount > cols - 1)
			{
				ccount = 0;
				rcount++;
			}

			matrix(rcount, ccount) = doubles[i];

			ccount++;

		}

	};

};

