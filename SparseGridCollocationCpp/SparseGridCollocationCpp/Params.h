#pragma once
using namespace Eigen;
using namespace std;

namespace Leicester
{
	namespace SparseGridCollocation
	{
		enum InitialMethod
		{
			MethodOfLines = 0,
			MonteCarlo = 1
		};

		class API Params
		{
		public:
			Params();
			~Params();
			double T;
			double Tdone;
			double Tend;
			double dt;
			double K;
			double r;
			double sigma;
			double theta;
			VectorXd inx1;
			VectorXd inx2;
			bool useCuda;
			InitialMethod GenerateSmoothInitialUsing;
			string SmoothInitialPath;
		};
	}
}

