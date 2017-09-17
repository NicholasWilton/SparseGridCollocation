#pragma once
#include "stdafx.h"

using Eigen::VectorXd;
namespace Leicester
{
	namespace SparseGridCollocation
	{
		class API SmoothInitialU
		{
		public:

			static VectorXd U();
			static VectorXd u;

		};
	}
}