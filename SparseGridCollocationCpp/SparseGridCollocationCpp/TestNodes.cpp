#include "stdafx.h"
#include "TestNodes.h"


Leicester::SparseGridCollocation::TestNodes::TestNodes()
{
}


Leicester::SparseGridCollocation::TestNodes::~TestNodes()
{
}

MatrixXd Leicester::SparseGridCollocation::TestNodes::GenerateTestNodes(VectorXd lowerLimits, VectorXd upperLimits, MatrixXd N, double coef)
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

MatrixXd Leicester::SparseGridCollocation::TestNodes::GenerateTestNodes(double timeLowerLimit, double timeUpperLimit, VectorXd lowerLimits, VectorXd upperLimits, MatrixXd N, double coef)
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
			linearDimension = VectorXd::LinSpaced(i, lowerLimits[n-1], upperLimits[n-1]);

		linearGrid.push_back(linearDimension);
	}


	MatrixXd TXYZ(product, N.cols());
	int dimension = 0;
	int dups = 1;
	for (auto linearVector : linearGrid)
	{
		//if (linearGrid.size() > dimension + 1)
			if (dimension == 0)
				dups = product / linearGrid[dimension].rows();
			else
				dups = dups / linearGrid[dimension].rows();

		TXYZ.col(dimension) = Replicate(linearVector, product, dups);
		dimension++;
	}

	return TXYZ;
}

vector<MatrixXd> Leicester::SparseGridCollocation::TestNodes::GenerateTestNodes(int nodes, VectorXd lowerLimits, VectorXd upperLimits, int dimensions)
{
	vector<VectorXd> linearGrid;
	int product = 1;
	double nthRoot = pow((double)nodes, (1.0 / dimensions) );
	int grid = floor(nthRoot);

	for (int n = 0; n < dimensions; n++)
	{
		product *= nthRoot;

		VectorXd linearDimension = VectorXd::LinSpaced(grid, lowerLimits[n], upperLimits[n]);

		linearGrid.push_back(linearDimension);
	}


	MatrixXd TXYZNodes(product, dimensions);
	MatrixXd TXYZGrid(grid, dimensions);
	int dimension = 0;

	for (auto linearVector : linearGrid)
	{
		int dups = 1;
		if (linearGrid.size() > dimension + 1)
			dups = linearGrid[dimension + 1].size();
		TXYZNodes.col(dimension) = Replicate(linearVector, product, dups);
		TXYZGrid.col(dimension) = linearVector;
		dimension++;
	}

	return { TXYZGrid, TXYZNodes };
}

MatrixXd Leicester::SparseGridCollocation::TestNodes::CartesianProduct(MatrixXd grid)
{
	int product = 1;
	int dimensions = grid.cols();

	for (int n = 0; n < dimensions; n++)
		product *= grid.rows();

	MatrixXd TXYZNodes(product, dimensions);
	
	for (int dimension = 0; dimension < dimensions; dimension++)
	{
		int dups = 1;
		if (grid.cols() > dimension + 1)
			dups = grid.col(dimension + 1).rows();
		TXYZNodes.col(dimension) = Replicate(grid.col(dimension), product, dups);
	
	}

	return TXYZNodes;
}

VectorXd Leicester::SparseGridCollocation::TestNodes::Replicate(VectorXd v, int totalLength, int dup)
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