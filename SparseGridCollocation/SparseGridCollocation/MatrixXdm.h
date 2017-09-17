#pragma once
#include "stdafx.h"
#include <iomanip> 
#include "Double.h" 
using Eigen::MatrixXd;
namespace Leicester
{
	namespace SparseGridCollocation
	{
		class MatrixXdM
		{
		private:
			MatrixXd m;

			//static double MatLabRounding(double d)
			//{
			//	return round(d * 1000000000000000) / 1000000000000000;
			//};

		public:
			MatrixXdM(MatrixXd v) : m(v) {}


			friend MatrixXdM operator*(MatrixXdM &a, MatrixXdM &b)
			{
				MatrixXd result(a.value().rows(), b.value().cols());
				//assume a & b are compatible
				for (int i = 0; i < result.rows(); i++)
					for (int j = 0; j < result.cols(); j++)
					{
						Double sum(0);
						for (int x = 0; x < a.value().cols(); x++)
							//for (int y = 0; y < b.value().rows(); y++)
						{
							double l = a.value()(i, x);
							double r = b.value()(x, j);
							Double m = Double(l) * Double(r);
							sum = sum + m;
							wstringstream ss;
							ss << setprecision(25) << "l=\t" << l << "\tr=\t" << r << "\tm=\t" << m.value() << "\t sum=\t" << sum.value() << endl;
							wstring w = ss.str();
							//Logger::WriteMessage(w.c_str());
						}
						result(i, j) = sum.value();
						//result(i,j) = Double(a(i, j)) * Double(b(i, j));
					}
				return MatrixXdM(result);

			};

			friend MatrixXdM operator*(double a, MatrixXdM &b)
			{
				MatrixXd result(b.value().rows(), b.value().cols());

				for (int i = 0; i < result.rows(); i++)
					for (int j = 0; j < result.cols(); j++)
					{
						Double sum(0);

						double r = b.value()(i, j);
						Double m = Double(a) * Double(r);

						wstringstream ss;
						ss << setprecision(25) << "l=\t" << a << "\tr=\t" << r << "\tm=\t" << m.value() << endl;
						wstring w = ss.str();
						//Logger::WriteMessage(w.c_str());

						result(i, j) = m.value();
					}
				return MatrixXdM(result);

			};

			friend MatrixXdM operator+(MatrixXdM &a, MatrixXdM &b)
			{
				MatrixXd result(b.value().rows(), b.value().cols());

				for (int i = 0; i < result.rows(); i++)
					for (int j = 0; j < result.cols(); j++)
					{
						Double sum(0);

						double r = b.value()(i, j);
						double l = a.value()(i, j);
						Double a = Double(l) + Double(r);

						wstringstream ss;
						ss << setprecision(25) << "l=\t" << l << "\tr=\t" << r << "\ta=\t" << a.value() << endl;
						wstring w = ss.str();
						//Logger::WriteMessage(w.c_str());

						result(i, j) = a.value();
					}
				return MatrixXdM(result);

			};

			friend MatrixXdM operator-(MatrixXdM &a, MatrixXdM &b)
			{
				MatrixXd result(b.value().rows(), b.value().cols());

				for (int i = 0; i < result.rows(); i++)
					for (int j = 0; j < result.cols(); j++)
					{
						Double sum(0);

						double r = b.value()(i, j);
						double l = a.value()(i, j);
						Double a = Double(l) - Double(r);

						wstringstream ss;
						ss << setprecision(25) << "l=\t" << l << "\tr=\t" << r << "\ta=\t" << a.value() << endl;
						wstring w = ss.str();
						//Logger::WriteMessage(w.c_str());

						result(i, j) = a.value();
					}
				return MatrixXdM(result);

			};

			MatrixXd value() const { return m; }
		};
	}
}