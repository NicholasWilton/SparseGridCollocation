#pragma once
#include "stdafx.h"

using Eigen::VectorXd;
namespace Leicester
{
	class API SmoothInitialX
	{
	public:

		static VectorXd X();
		static VectorXd x;

	};
}