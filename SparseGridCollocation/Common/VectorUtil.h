#pragma once

using namespace Eigen;
using namespace std;

class API VectorUtil
{
public:
	VectorUtil();
	~VectorUtil();
	static VectorXd Select(VectorXd A, double notEqual);
	static VectorXd PushAndQueue(double push, VectorXd A, double queue);
	static VectorXd Queue(VectorXd A, double queue);
	static VectorXd Push(VectorXd A, double push);
	static VectorXd Diff(VectorXd A);
};

