#pragma once

#include "stdafx.h"
#include <algorithm>
#include <vector>

// Simple little templated comparison functor
template <typename MatrixT>
bool compareRows(MatrixT a, MatrixT b) {
	return a(0, 0) < b(0, 0);
}

// These are the 6 template arguments to every Eigen matrix
template <typename Scalar, int rows, int cols, int options, int maxRows, int maxCols>
Eigen::Matrix<Scalar, rows, cols, options, maxRows, maxCols> sortMatrix(
	Eigen::Matrix<Scalar, rows, cols, options, maxRows, maxCols> target
) {
	// Manually construct a vector of correctly-typed matrix rows
	std::vector<Eigen::Matrix<Scalar, 1, cols>> matrixRows;
	for (unsigned int i = 0; i < target.rows(); i++)
		matrixRows.push_back(target.row(i));
	std::sort(
		matrixRows.begin(),
		matrixRows.end(),
		compareRows<Eigen::Matrix<Scalar, 1, cols>>
	);

	Eigen::Matrix<Scalar, rows, cols, options, maxRows, maxCols> sorted(target.rows(), target.cols());
	for (unsigned int i = 0; i < matrixRows.size(); i++)
		sorted.row(i) = matrixRows[i];
	return sorted;
}
