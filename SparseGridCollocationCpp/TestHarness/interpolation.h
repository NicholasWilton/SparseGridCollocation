#pragma once


#include "stdafx.h"

using namespace Eigen;
using namespace std;

class Interpolation {
public:
	static map<string, vector<vector<MatrixXd>>> GetInterpolation();
};