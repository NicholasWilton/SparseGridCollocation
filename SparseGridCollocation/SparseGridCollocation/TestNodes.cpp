#include "stdafx.h"
#include "TestNodes.h"


Leicester::TestNodes::TestNodes()
{
}


Leicester::TestNodes::~TestNodes()
{
}

MatrixXd Leicester::TestNodes::GenerateTestNodes(VectorXd lowerLimits, VectorXd upperLimits, MatrixXd N, double coef)
{
	vector<VectorXd> linearGrid;
	int product = 1;
	
	for (int n = 0; n < N.cols(); n++) //N.Cols() is #dimensions
	{
		int i = N(0, n);
		product *= i;
		VectorXd linearDimension = VectorXd::LinSpaced(i, lowerLimits[n], upperLimits[n]);
		linearGrid.push_back(linearDimension);
	}

	MatrixXd TXYZ(product, N.cols());
	int dimension = 0;

	for (auto linearVector : linearGrid)
	{
		int dups = 1;
		if (linearGrid.size() > dimension + 1)
			dups = linearGrid[dimension + 1].size();
		TXYZ.col(dimension) = Replicate(linearVector, product, dups);
		dimension++;
	}

	return TXYZ;
}

MatrixXd Leicester::TestNodes::GenerateTestNodes(double timeLowerLimit, double timeUpperLimit, VectorXd lowerLimits, VectorXd upperLimits, MatrixXd N, double coef)
{
	vector<VectorXd> linearGrid;
	int product = 1;
	
	for (int n = 0; n < N.cols(); n++) //N.Cols() is #dimensions
	{
		int i = N(0, n);
		product *= i;
	
		VectorXd linearDimension;
		if (n == 0)
			linearDimension = VectorXd::LinSpaced(i, timeLowerLimit, timeUpperLimit);
		else
			linearDimension = VectorXd::LinSpaced(i, lowerLimits(0, n), upperLimits(0, n));

		linearGrid.push_back(linearDimension);
	}


	MatrixXd TXYZ(product, N.cols());
	int dimension = 0;

	for (auto linearVector : linearGrid)
	{
		int dups = 1;
		if (linearGrid.size() > dimension + 1)
			dups = linearGrid[dimension + 1].size();
		TXYZ.col(dimension) = Replicate(linearVector, product, dups);
		dimension++;
	}

	return TXYZ;
}

MatrixXd Leicester::TestNodes::GenerateTestNodes(int nodes, VectorXd lowerLimits, VectorXd upperLimits, int dimensions)
{
	vector<VectorXd> linearGrid;
	int product = 1;

	for (int n = 0; n < dimensions; n++)
	{
		product *= nodes;

		VectorXd linearDimension = VectorXd::LinSpaced(nodes, lowerLimits[n], upperLimits[n]);

		linearGrid.push_back(linearDimension);
	}


	MatrixXd TXYZ(product, dimensions);
	int dimension = 0;

	for (auto linearVector : linearGrid)
	{
		int dups = 1;
		if (linearGrid.size() > dimension + 1)
			dups = linearGrid[dimension + 1].size();
		TXYZ.col(dimension) = Replicate(linearVector, product, dups);
		dimension++;
	}

	return TXYZ;
}

VectorXd Leicester::TestNodes::Replicate(VectorXd v, int totalLength, int dup)
{
	VectorXd Result(totalLength);

	for (int i = 0; i < totalLength; i += (v.size() * dup))
	{
		for (int j = 0; j < v.size(); j++)
		{
			for (int duplicated = 0; duplicated < dup; duplicated++)
			{
				int idx = i + (j * dup) + duplicated;
				if (idx < totalLength)
				{
					Result[idx] = v[j];
					//cout << "idx="<< idx << " v[j]=" << v[j] << endl;
				}

			}
		}
	}
	return Result;
}