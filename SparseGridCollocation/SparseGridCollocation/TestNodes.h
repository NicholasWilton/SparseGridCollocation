#pragma once
using namespace Eigen;
using namespace std;

namespace Leicester
{
	class API TestNodes
	{
	public:
		TestNodes();
		~TestNodes();

		static MatrixXd GenerateTestNodes(VectorXd lowerLimits, VectorXd upperLimits, MatrixXd N, double coef);
		static vector<MatrixXd> GenerateTestNodes(int nodes, VectorXd lowerLimits, VectorXd upperLimits, int dimensions);
		static MatrixXd GenerateTestNodes(double timeLowerLimit, double timeUpperLimit, VectorXd lowerLimits, VectorXd upperLimits, MatrixXd N, double coef);
		static MatrixXd CartesianProduct(MatrixXd grid);
		static VectorXd Replicate(VectorXd v, int totalLength, int dup);

	};
}
