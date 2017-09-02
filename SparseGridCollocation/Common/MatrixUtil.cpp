#include "stdafx.h"
#include "MatrixUtil.h"


MatrixUtil::MatrixUtil()
{
}


MatrixUtil::~MatrixUtil()
{
}

MatrixXd MatrixUtil::Diff(MatrixXd m)
{
	MatrixXd result(m.rows() - 1, m.cols());
	for (int j = 0; j < m.cols(); j++)
	for (int i = 0; i < m.rows() - 1; i++)
	{
		result(i,j) = m(i + 1, j) - m(i, j);
	}
	return result;
}

MatrixXd MatrixUtil::PushAndQueueRows(double push , MatrixXd A, double queue)
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

MatrixXd MatrixUtil::PushRows(MatrixXd A, double push)
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

MatrixXd MatrixUtil::QueueRows(MatrixXd A, double queue)
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

MatrixXd MatrixUtil::Select(MatrixXd A, double notEqual)
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