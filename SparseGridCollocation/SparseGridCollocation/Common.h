#pragma once
using namespace Eigen;
using namespace std;

class API Common
{
public:
	Common();
	~Common();
	static wstring Common::printMatrix(MatrixXd m);
};

