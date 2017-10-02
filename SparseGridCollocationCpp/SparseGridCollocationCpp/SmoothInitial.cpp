#include "stdafx.h"
#include "SmoothInitial.h"

Leicester::SparseGridCollocation::SmoothInitial::SmoothInitial()
{}

Leicester::SparseGridCollocation::SmoothInitial::SmoothInitial(double T, MatrixXd TestNodes, VectorXd Lambda, VectorXd C)
{
	this->T = T;
	this->TestNodes = TestNodes;
	this->Lambda = Lambda;
	this->C = C;
}


Leicester::SparseGridCollocation::SmoothInitial::~SmoothInitial()
{
}
