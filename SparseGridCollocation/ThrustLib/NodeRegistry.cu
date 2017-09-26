#include "NodeRegistry.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <thrust/random.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>

using namespace Leicester::ThrustLib;
using namespace std;

namespace Leicester
{
	namespace ThrustLib
	{

		__device__
		void subnumber(int b, int d, double matrix[])
		{

			double *L = NULL;
			if (d == 1)
			{
				double * l = (double*)malloc(3 * sizeof(double));
				l[0] = 1;
				l[1] = 1;
				l[2] = b;
				L = l;
			}
			else
			{
				int nbot = 1;

				int Lrows = 0;
				int Lcols = 0;
				for (int i = 0; i < b - d + 1; i++)
				{
					double* indextemp = (double*)malloc(512 * sizeof(double));

					subnumber(b - (i + 1), d - 1, indextemp);
					//printMatrix_CUDA(indextemp, dim3(indextemp[0], indextemp[1]));

					int s = indextemp[0];
					int ntop = nbot + s - 1;

					double*l = (double*)malloc((ntop*d + 2) * sizeof(double));

					l[0] = ntop;
					l[1] = d;
					double *ones = (double*)malloc((s + 2) * sizeof(double));
					ones[0] = s;
					ones[1] = 1;

					thrust::fill(thrust::seq, ones + 2, ones + 2 + s, (i + 1));

					int start = nbot;
					int end = start + ntop - nbot;

					//fill the first column with 'ones'
					//thrust::fill(thrust::seq, l + 2 + start, l + 2 + end, (i + 1));
					//fill the rest with 'indextemp'
					//thrust::copy(thrust::seq, indextemp + 2, indextemp + 2 + (int)(l[0] * l[1]) - 1, l + start + ntop);
					int jMin = 0;
					int increment = 1;
					if (L != NULL)
					{
						int count = 0;
						for (int x = 0; x < L[1]; x++)
							for (int y = 0; y < L[0]; y++)
							{
								int diff = l[0] - L[0];
								l[count + 2 + x * diff] = L[count + 2];
								//int indx = (x * L[0]) + y + 2;

								//l[indx] = L[indx];
								count++;
							}
						jMin = L[0];
						increment = L[0];
					}

					int rows = l[0];
					int cols = l[1];
					int k = 0;
					for (int j = jMin; j < rows * cols; j += rows, k++)
					{
						int indx = j + 2;
						if (j - jMin < rows)//first col
							l[indx] = i + 1;
						else
							l[indx] = indextemp[k + 1];
					}

					nbot = ntop + 1;

					//if (Lrows > 0)
					//{
					//	thrust::copy(thrust::seq, L, L + (Lrows * Lcols) - 1, l);
					//}
					L = (double*)malloc(sizeof(double) * (l[0] * l[1] + 2));
					for (int i = 0; i < (int)(l[0] * l[1]) + 2; i++)
						L[i] = l[i];
					Lrows = ntop;
					Lcols = d;
				}
			}
			for (int i = 0; i < (int)(L[0] * L[1]) + 2; i++)
				matrix[i] = L[i];

		}

		__device__
		void GetN(int b, int d, double N[])
		{
			double *d_L = (double*)malloc(3 * sizeof(double));
			d_L[0] = 1;
			d_L[1] = 1;

			subnumber(b, d, d_L);
			int ch = d_L[0];
			//free(N);

			//N = (double*)malloc((2 + ch * d) * sizeof(double));
			N[0] = ch;
			N[1] = d;
			int idx = 2;
			for (int i = 0; i < ch; i++)
				for (int j = 0; j < d; j++, idx++)
				{
					//int idx = 2 + j + (i * ch);
					N[idx] = pow((double)2, d_L[idx]) + 1;
					//for ()
				}

		}
		
		__device__
		double* GetColumn(double matrix[], int col)
		{
			double* result = (double*)malloc((2 + matrix[0]) * sizeof(double));
			int columnStart = 2 + ((matrix[0] + 2) * col);
			result[0] = matrix[columnStart];
			result[1] = 1;
			for (int i = 0; i < result[0]; i++)
			{
				int idx = i + 2 + columnStart;
				result[i + 2] = matrix[idx];
			}
			return result;
		}

		__device__
		void SetColumn(double matrix[], double vector[], int col)
		{
			/*double* result = (double*)malloc((2 + matrix[0] * matrix[1]) * sizeof(double));
			result[0] = matrix[0];
			result[1] = matrix[1];*/
			for (int i = 0; i < vector[0]; i++)
			{
				int idx = i + (matrix[0] * col);
				matrix[idx + 2] = vector[i + 2];
			}
			//return result;
		}

		__device__
		double* GetRow(double matrix[], int row)
		{
			double* result = (double*)malloc((2 + matrix[1]) * sizeof(double));
			result[0] = 1;
			result[1] = matrix[1];
			int rowIdx = 0;
			for (int i = 0; i < matrix[0] * matrix[1]; i++)
			{
				if ((i % (int)matrix[0]) == row)
				{
					result[rowIdx + 2] = matrix[i + 2];
					rowIdx++;
				}
			}
			return result;
		}

		__device__
		double* ReplicateN(double linearVector[], double totalLength, int dups)
		{
			double* Result = (double*)malloc((2 + totalLength) * sizeof(double));
			Result[0] = totalLength;
			Result[1] = 1;
			int size = linearVector[0] * linearVector[1];
			for (int i = 0; i < totalLength; i += (size * dups))
			{
				for (int j = 0; j < size; j++)
				{
					for (int duplicated = 0; duplicated < dups; duplicated++)
					{
						int idx = i + (j * dups) + duplicated;
						if (idx < totalLength)
						{
							Result[idx + 2] = linearVector[j + 2];
							//cout << "idx="<< idx << " v[j]=" << v[j] << endl;
						}

					}
				}
			}
			return Result;
		}

		__device__
		double Max(double matrix[])
		{
			double max = -1;
			double length = 2 + matrix[0] * matrix[1];
			for (int i = 2; i < length; i++)
				if (matrix[i] > max)
					max = matrix[i];
			return max;
		}

		__device__
		double* VectorLinSpaced(int i, double lowerLimit, double upperLimit)
		{
			double difference = upperLimit - lowerLimit;
			double dx = difference / (i - 1);
			double* result = (double*)malloc((i + 2) * sizeof(double));
			result[0] = i;
			result[1] = 1;
			for (int j = 0; j < i; j++)
			{
				result[j + 2] = lowerLimit + (j * dx);
			}
			return result;
		}

		__device__
		double* GenerateTestNodes(double timeLowerLimit, double timeUpperLimit, double lowerLimits[], double upperLimits[], double N[])
		{
			//vector<VectorXd> linearGrid;

			int product = 1;
			int nCols = N[1];
			int nRows = N[0];
			double max = Max(N);
			double* linearGrid = (double*)malloc((2 + (nCols * (2 + max))) * sizeof(double)); // this is an array of NCols vectors of max-rows each
			linearGrid[0] = max;
			linearGrid[1] = nCols;
			for (int n = 0; n < nCols; n++) //N.Cols() is #dimensions
			{
				int idx = n * nRows;
				int i = N[idx + 2];
				product *= i;

				//VectorXd linearDimension;
				double* linearDimension;
				if (n == 0)
					linearDimension = VectorLinSpaced(i, timeLowerLimit, timeUpperLimit);
				else
					linearDimension = VectorLinSpaced(i, lowerLimits[n - 1], upperLimits[n - 1]);
				double length = linearDimension[0] * linearDimension[1] + 2;
				//linearGrid.push_back(linearDimension);
				for (int j = 0; j < length; j++)
				{
					int idx = 2 + n* (linearGrid[0] + 2);
					linearGrid[idx + j] = linearDimension[j];
				}

			}


			//MatrixXd TXYZ(product, N.cols());
			double* TXYZ = (double*)malloc(product * nCols * sizeof(double));
			TXYZ[0] = product;
			TXYZ[1] = nCols;
			int dimension = 0;
			int dups = 1;
			for (int col = 0; col < nCols; col++)
			{
				int idx = 2 + dimension * (2 + linearGrid[0]);
				if (dimension == 0)
				{
					dups = product / linearGrid[idx];
				}
				else
					dups = dups / linearGrid[idx];
				double* linearVector = GetColumn(linearGrid, col);
				double* column = ReplicateN(linearVector, product, dups);
				SetColumn(TXYZ, column, col);
				dimension++;
			}
			return TXYZ;
		}

		__global__
			void GenerateTestNodes(double timeLowerLimit, double timeUpperLimit, double N[], double* TXYZ)
		{
			double lower[1] = { 0 };
			double upper[1] = { 300 };
			double n[4] = { 1,2,N[0],N[1] };
			double* result = GenerateTestNodes(timeLowerLimit, timeUpperLimit, lower, upper, n);
			for (int i =0; i < 2 + (2 * N[2] * N[3]); i ++)
				TXYZ[i] = result[i];
		}

		__global__ void BuildRegistry(int b, int d, double tLower, double tUpper, double xLower, double xHigher, double** nodes)
		{
			double* N = (double*)malloc(512 * sizeof(double));
			GetN(b, d, N);
			int rows = N[0];
			double n[] = { rows };
			nodes[0] = n;
			for (int i = 1; i <= rows; i++)
			{
				double* n = GetRow(N, 0);
				double lower[1] = { 0 };
				double upper[1] = { 300 };
				double* TXYZ = GenerateTestNodes(tLower, tUpper, lower, upper, n);
				nodes[i] = TXYZ;
			}
		}
	}
}