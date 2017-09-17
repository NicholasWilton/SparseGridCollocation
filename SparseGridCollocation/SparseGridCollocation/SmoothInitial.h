#pragma once
#include "stdafx.h"
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API SmoothInitial
		{
		public:
			SmoothInitial();
			SmoothInitial(double T, MatrixXd TestNodes, VectorXd Lambda, VectorXd C);
			~SmoothInitial();
			double T;
			MatrixXd TestNodes;
			VectorXd Lambda;
			MatrixXd C;
			VectorXd S;
			VectorXd U;
		};
	}
}
