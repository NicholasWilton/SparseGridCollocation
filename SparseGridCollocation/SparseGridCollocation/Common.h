#pragma once
using namespace Eigen;
using namespace std;


class API Common
{

public:
	
	Common();
	~Common();
	static wstring Common::printMatrix(MatrixXd m);
	static double Common::prod(vector<double> x);
	static vector<double> Common::linspace(double a, double b, size_t N);
};

