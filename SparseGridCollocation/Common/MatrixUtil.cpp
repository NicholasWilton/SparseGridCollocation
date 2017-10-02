#include "stdafx.h"
#include "MatrixUtil.h"


VectorXd Leicester::Common::MatrixUtil::Diff(MatrixXd m)
{
	VectorXd result(m.rows() - 1);
	for (int i = 0; i < m.rows() - 1; i++)
	{
		double sum = 0;
		for (int j = 0; j < m.cols(); j++)
		{
			sum += (m(i + 1, j) - m(i, j)) * (m(i + 1, j) - m(i, j));
		}
		result[i] = sqrt(sum);
	}
	return result;
}

MatrixXd Leicester::Common::MatrixUtil::PushAndQueueRows(double push , MatrixXd A, double queue)
{
	MatrixXd result(A.rows() + 2, A.cols());

	for (int j = 0; j < A.cols(); j++)
	{
		result(0, j) = push;
		for (int i = 1; i < A.rows(); i++)
		{
			result(i, j) = A(i - 1, j);
		}
		result(A.rows(), j) = queue;
	}
	return result;

}

MatrixXd Leicester::Common::MatrixUtil::PushRows(MatrixXd A, double push)
{
	MatrixXd result(A.rows() + 1, A.cols());
	
	for (int j = 0; j < A.cols(); j++)
	{
		result(0,j) = push;
		for (int i = 1; i <= A.rows(); i++)
		{
			result(i, j) = A(i - 1, j);
		}
	}
	return result;

}

MatrixXd Leicester::Common::MatrixUtil::QueueRows(MatrixXd A, double queue)
{
	MatrixXd result(A.rows() + 1, A.cols());

	for (int j = 0; j < A.cols(); j++)
	{
		for (int i = 0; i < A.rows(); i++)
		{
			result(i,j) = A(i, j);
		}
		result(A.rows(), j) = queue;
	}
	return result;
}

MatrixXd Leicester::Common::MatrixUtil::Select(MatrixXd A, double notEqual)
{
	
	vector<vector<double>> inter;
	
	for (int j = 0; j < A.cols(); j++)
	{
		vector<double> v;
		for (int i = 0; i < A.rows(); i++)
		{
			if (A(i, j) != notEqual)
				v.push_back(A(i, j));
		}
		inter.push_back(v);
	}
	//should be square
	MatrixXd result(inter[0].size(), inter.size());

	int y = 0;
	for (auto v : inter)
	{
		int x = 0;
		for (auto d : v)
		{
			result(x,y) = d;
			x++;
		}
		y++;
	}
	return result;
}