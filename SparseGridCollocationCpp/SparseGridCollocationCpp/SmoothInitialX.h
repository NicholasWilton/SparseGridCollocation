#pragma once
#include "stdafx.h"

using Eigen::VectorXd;
namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API SmoothInitialX
		{
		public:

			static VectorXd X();
			static VectorXd x;

		};
	}
}