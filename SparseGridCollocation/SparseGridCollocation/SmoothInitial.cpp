#include "stdafx.h"
#include "SmoothInitial.h"

Leicester::SmoothInitial::SmoothInitial()
{}

Leicester::SmoothInitial::SmoothInitial(double T, MatrixXd TestNodes, VectorXd Lambda, VectorXd C)
{
	this->T = T;
	this->TestNodes = TestNodes;
	this->Lambda = Lambda;
	this->C = C;
}


Leicester::SmoothInitial::~SmoothInitial()
{
}
