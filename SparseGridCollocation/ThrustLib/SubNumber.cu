#include "SubNumber.h"

namespace Leicester
{
	namespace ThrustLib
	{
		//struct N_functor
		//{
		//	N_functor() {}
		//	__device__ double operator()(double L)
		//	{
		//		return 
		//	}
		//};

		//__device__ void printMatrix_CUDA(double *matrix, dim3 dimMatrix)
		//{

		//	printf("printing matrix data=");
		//	for (int x = 0; x < 2 + dimMatrix.x * dimMatrix.y; x++)
		//		printf("%f,", matrix[x]);
		//	printf("\r\n");
		//	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);

		//	for (int y = 0; y < dimMatrix.y; y++)
		//	{
		//		for (int x = 0; x < dimMatrix.x; x++)
		//		{
		//			//int idx = (y * dimMatrix.x) + x;
		//			int idx = (x * dimMatrix.y) + y;
		//			//if ( mSize > idx)
		//			printf("indx=%i value=%16.10f\t", idx, matrix[idx + 2]);
		//		}
		//		printf("\r\n");
		//	}

		//}

		//__global__ void Add_CUDA(int b, int d, double *N)
		//{
		//	double *d_L = (double*)malloc(3 * sizeof(double));
		//	d_L[0] = 1;
		//	d_L[1] = 1;

		//	subnumber(b, d, d_L);
		//	int ch = d_L[0];
		//	free(N);
		//	//N = (double*)malloc((2 + (L[0] * L[1])) * sizeof(double));
		//	N = (double*)malloc( (2 + ch * d) * sizeof(double));
		//	for (int i = 0; i < ch; i++)
		//		for (int j = 0; j < d; j++)
		//		{
		//			int idx = j + (i * ch);
		//			N[idx] = (d_L[idx] * d_L[idx]) + 1;
		//		}
		//	
		//	//thrust::transform(thrust::seq, L + 2, L + 2 + (int)(L[0] * L[1]), N,  )
		//}

		//struct subNumber_functor
		//{
		//	int b;
		//	int d;
		//	subNumber_functor(int _b, int _d, int &_nbot) { b = _b; d = _d; }
		//	int nbot;

		//	template<typename Tuple> __device__ void operator()(Tuple t)
		//	{
		//		int i = (int)thrust::get<0>(t);
		//		double* temp;
		//		matrixDim m = subnumber(b - (i + 1), d - 1, temp);
		//		int s = m.rows;
		//		int ntop = nbot + s - 1;
		//		//double* l = (double*)malloc(ntop * d * sizeof(double));
		//		device_vector<double> l(ntop * d);
		//		//double* ones = (double*)malloc(s * 1 * sizeof(double));
		//		device_vector<double> ones(s);
		//		thrust::fill(ones.begin(), ones.end(), 1);
		//		thrust::copy_n(ones.begin(), s, l.begin());
		//		thrust::transform(ones.begin(), ones.end(), l.begin(), vectorScalarMultiply(i + 1));


		//		nbot = ntop + 1;
		//	}
		//};
		//__device__ double* indextemp;
		
		//__device__ int getOne()
		//{
		//	return 1;
		//}

	/*	__device__ void subnumber(int b, int d, double *matrix)
		{
			
			double *L;
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
					double* indextemp = (double*)malloc(3 * sizeof(double));
				
					subnumber(b - (i + 1), d - 1, indextemp);
					printMatrix_CUDA(indextemp, dim3(indextemp[0], indextemp[1]));
					
					int s = indextemp[0];
					int ntop = nbot + s - 1;
					
					double*l = (double*)malloc(ntop*d * sizeof(double) + 2);
					
					l[0] = ntop;
					l[1] = d;
					double *ones = (double*)malloc(s * sizeof(double));
					ones[0] = s;
					ones[1] = 1;

					thrust::fill(thrust::seq, ones + 1, ones + 2 + s, (i + 1));

					int start = nbot - 1;
					int end = start + ntop - nbot + 1;

					thrust::fill(thrust::seq, l + 2 + start, l +2 + end, (i + 1));
					thrust::copy(thrust::seq, indextemp+ 2, indextemp + 2 + (int)(l[0] * l[1]) -1, l + start + ntop);
					
					nbot = ntop + 1;

					if (Lrows > 0)
					{
						thrust::copy(thrust::seq, L, L + (Lrows * Lcols) -1, l);
					}
					L = l;
					Lrows = ntop;
					Lcols = d;
				}
			}
			matrix = L;

		}*/
	}
}