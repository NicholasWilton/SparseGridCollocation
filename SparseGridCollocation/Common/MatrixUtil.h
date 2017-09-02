#pragma once
using namespace Eigen;
using namespace std;

class API MatrixUtil
{
public:
	MatrixUtil();
	~MatrixUtil();
	static MatrixXd Diff(MatrixXd m);
	static MatrixXd PushRows(MatrixXd A, double push);
	static MatrixXd QueueRows(MatrixXd A, double queue);
	static MatrixXd PushAndQueueRows(double push, MatrixXd A, double queue);
	static MatrixXd Select(MatrixXd A, double notEqual);
};

