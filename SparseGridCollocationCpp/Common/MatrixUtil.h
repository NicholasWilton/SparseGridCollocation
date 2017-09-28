#pragma once
using namespace Eigen;
using namespace std;

namespace Leicester
{
	namespace Common
	{
		class API MatrixUtil
		{
		public:
			static VectorXd Diff(MatrixXd m);
			static MatrixXd PushRows(MatrixXd A, double push);
			static MatrixXd QueueRows(MatrixXd A, double queue);
			static MatrixXd PushAndQueueRows(double push, MatrixXd A, double queue);
			static MatrixXd Select(MatrixXd A, double notEqual);
		};
	}
}
