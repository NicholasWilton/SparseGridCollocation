#include "stdafx.h"
#include "Common.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace Eigen;
using namespace std;

Common::Common()
{
}


Common::~Common()
{
}

wstring Common::printMatrix(MatrixXd m)
{
	int cols = m.cols();
	int rows = m.rows();

	const IOFormat fmt(2, DontAlignCols, "\t", " ", "", "", "", "");

	wstringstream ss;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			ss << m(i, j) << "\t";

		}
		ss << "\r\n";
	}

	return ss.str();
}