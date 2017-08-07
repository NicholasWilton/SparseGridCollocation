#include "stdafx.h"
#include "VectorUtil.h"

using namespace Eigen;
using namespace std;


VectorUtil::VectorUtil()
{
}


VectorUtil::~VectorUtil()
{
}


VectorXd VectorUtil::Diff(VectorXd A)
{
	VectorXd result(A.rows() - 1);
	for (int i = 0; i < A.rows() - 1; i++)
	{
		result[i] = A[i + 1] - A[i];
	}
	return result;

}

VectorXd VectorUtil::Push(VectorXd A, double push)
{
	VectorXd result(A.rows() + 1);
	result[0] = push;
	for (int i = 1; i <= A.rows(); i++)
	{
		result[i] = A[i - 1];
	}

	return result;

}

VectorXd VectorUtil::Queue(VectorXd A, double queue)
{
	VectorXd result(A.rows() + 1);

	for (int i = 0; i < A.rows(); i++)
	{
		result[i] = A[i];
	}
	result[A.rows()] = queue;
	return result;
}

VectorXd VectorUtil::PushAndQueue(double push, VectorXd A, double queue)
{
	VectorXd result(A.rows() + 2);
	result[0] = push;
	for (int i = 0; i <= A.rows(); i++)
	{
		result[i + 1] = A[i];
	}
	result[A.rows() + 1] = queue;
	return result;
}

VectorXd VectorUtil::Select(VectorXd A, double notEqual)
{
	vector<double> inter;
	for (int i = 0; i < A.rows(); i++)
	{
		if (A[i] != notEqual)
			inter.push_back(A[i]);
	}
	VectorXd result(inter.size());
	int count = 0;
	for (auto i : inter)
	{
		result[count] = i;
		count++;
	}
	return result;
}