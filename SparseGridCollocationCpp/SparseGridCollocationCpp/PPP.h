#pragma once
#include "stdafx.h"

using Eigen::RowVectorXd;

namespace Leicester
{
	namespace SparseGridCollocation
	{
		class PPP
		{
		public:

			static double Calculate(const RowVectorXd &X);
			//static VectorXd U();
			//static VectorXd X();
		};
	}
}