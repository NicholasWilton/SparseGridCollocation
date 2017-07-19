#include "stdafx.h"
#include "CppUnitTest.h"
#include "SparseGridCollocation.h"
#include "Interpolation.h"
#include "windows.h"
#include "testCommon.h"
#include "Common.h"
#include "Test.h"
#include "MatrixXdm.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "../include/boost_1_64_0/boost/multiprecision/cpp_dec_float.hpp"



using Eigen::Matrix2d;
using namespace boost::multiprecision;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;



namespace UnitTest
{
	TEST_CLASS(testInterpolation)
	{
		vector<vector<MatrixXd>> Get3()
		{
			vector<MatrixXd> uLamb3;
			//MatrixXd l1(15, 1);
			//l1 << 168.177040362907, -384.857133569469, 437.285285931294, -355.212798882940, 443.385219414521, -109.178371522743, 221.964431301908, -185.896683167106, 91.7816691444462, -271.563020925667, 179.474487920222, -412.630669834898, 475.510735359416, -393.047823887079, 457.473155903381;
			//uLamb3.push_back(l1);
			//MatrixXd l2(15, 1);
			//l2 << 266.292960914272, -700.744331062505, 759.616443800679, -184.485437051410, 574.752360107266, -715.267344742912, 324.264851093682, -938.943872118302, 1106.30094053661, -187.682488571149, 578.284311734800, -714.053999346216, 276.596980754002, -715.471078725550, 762.783738957473;

			MatrixXd l1 = LoadData("inner_testLamb3_1.txt", 15, 15, 1);
			MatrixXd l2 = LoadData("inner_testLamb3_2.txt", 15, 15, 1);

			uLamb3 = {l1,l2};

			MatrixXd TX1 = MatrixXd(15, 2);
			TX1 << 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.4325, 0, 0.4325, 75, 0.4325, 150, 0.4325, 225, 0.4325, 300, 0.865, 0, 0.865, 75, 0.865, 150, 0.865, 225, 0.865, 300;
			MatrixXd TX2 = MatrixXd(15, 2);
			TX2 << 0, 0, 0, 150, 0, 300, 0.216250000000000, 0, 0.216250000000000, 150, 0.216250000000000, 300, 0.432500000000000, 0, 0.432500000000000, 150, 0.432500000000000, 300, 0.648750000000000, 0, 0.648750000000000, 150, 0.648750000000000, 300, 0.865000000000000, 0, 0.865000000000000, 150, 0.865000000000000, 300;
			vector<MatrixXd> uTX = { TX1, TX2 };

			MatrixXd c1(1, 2);
			c1 << 1.73000000000000, 600;
			MatrixXd c2(1, 2);
			c2 << 1.73000000000000, 600;
			vector<MatrixXd> uC = { c1, c2 };

			MatrixXd a1(1, 2);
			a1 << 2, 4;
			MatrixXd a2(1, 2);
			a2 << 4, 2;
			vector<MatrixXd> uA = { a1, a2 };
			vector<vector<MatrixXd>> third = { uLamb3, uTX, uC, uA };
			return third;
		};

		vector<vector<MatrixXd>> Get2()
		{
			vector<MatrixXd> uLamb3;
			/*MatrixXd l1(9, 1);
			l1 << 273.169921181606,				- 696.599817216584,				732.148991790275,				- 191.889255272131,				521.665751945236,				- 582.757329851410,				284.102691920217,				- 712.786022564190,				736.696836746533;
			*/
			MatrixXd l1 = LoadData("inner_testLamb2_1.txt", 9, 9, 1);
			uLamb3.push_back(l1);

			MatrixXd TX1 = MatrixXd(9, 2);
			TX1 << 0, 0, 0, 150, 0, 300, 0.432500000000000, 0, 0.432500000000000, 150, 0.432500000000000, 300, 0.865000000000000, 0, 0.865000000000000, 150, 0.865000000000000, 300;
			vector<MatrixXd> uTX = { TX1 };

			MatrixXd c1(1, 2);
			c1 << 1.73000000000000, 600;
			vector<MatrixXd> uC = { c1 };

			MatrixXd a1(1, 2);
			a1 << 2,	2;
			vector<MatrixXd> uA = { a1};
			vector<vector<MatrixXd>> third = { uLamb3, uTX, uC, uA };
			return third;

		};

		vector<vector<MatrixXd>> Get_3()
		{

			/*MatrixXd l1(15, 1);
			l1 << -42.8498877775422, 141.016331069450, -249.998444271759, 258.615502944090, -123.518857109165, 65.0489756228674, -219.762990596407, 397.963768108779, -417.194722894546, 200.480323513864, -34.8966003832208, 119.274580096719, -217.965206264739, 229.772250182186, -110.694184815550;
			MatrixXd l2(15, 1);
			l2 << -0.0568069765069908, 0.0997753173328967, -0.0568069764883387, 0.0369586729170275, -0.0649139162828327, 0.0369586728689974, -0.0138051312405887, 0.0242472216107991, -0.0138051311798996, -0.0763568261614475, 0.134112516310045, -0.0763568262085483, 0.0616904205757057, -0.108352559302475, 0.0616904205955072;
			*/

			MatrixXd l1 = LoadData("inner_testLamb_3_1.txt", 15, 15, 1);
			MatrixXd l2 = LoadData("inner_testLamb_3_2.txt", 15, 15, 1);
			vector<MatrixXd> uLamb3{ l1,l2 };

			MatrixXd tx1(15, 2);
			tx1 << 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.432500000000000, 0, 0.432500000000000, 75, 0.432500000000000, 150, 0.432500000000000, 225, 0.432500000000000, 300, 0.865000000000000, 0, 0.865000000000000, 75, 0.865000000000000, 150, 0.865000000000000, 225, 0.865000000000000, 300;
			MatrixXd tx2(15,2);
			tx2 << 0, 0, 0, 150, 0, 300, 0.216250000000000, 0, 0.216250000000000, 150, 0.216250000000000, 300, 0.432500000000000, 0, 0.432500000000000, 150, 0.432500000000000, 300, 0.648750000000000, 0, 0.648750000000000, 150, 0.648750000000000, 300, 0.865000000000000, 0, 0.865000000000000, 150, 0.865000000000000, 300;
			vector<MatrixXd> uTX{ tx1,tx2 };

			MatrixXd c1(1, 2);
			c1 << 1.73000000000000, 600;
			MatrixXd c2(1, 2);
			c2 << 1.73000000000000, 600;
			vector<MatrixXd> uC = { c1, c2 };

			MatrixXd a1(1, 2);
			a1 << 2, 4;
			MatrixXd a2(1, 2);
			a2 << 4, 2;
			vector<MatrixXd> uA = { a1, a2 };

			vector<vector<MatrixXd>> third = { uLamb3, uTX, uC, uA };
			return third;
		}

		vector<vector<MatrixXd>> Get4()
		{
			MatrixXd l1(27, 1);
			Load("lamb4_1.txt", l1);
			MatrixXd l2(25, 1);
			Load("lamb4_2.txt", l2);
			MatrixXd l3(27, 1);
			Load("lamb4_3.txt", l3);
			vector<MatrixXd> uLamb4{ l1,l2,l3 };

			MatrixXd tx1(27, 2);
			Load("TX4_1.txt", tx1);
			MatrixXd tx2(25, 2);
			Load("TX4_2.txt", tx2);
			MatrixXd tx3(27, 2);
			Load("TX4_3.txt", tx3);
			vector<MatrixXd> uTX{ tx1,tx2, tx3 };

			MatrixXd c1(1, 2);
			c1 << 1.73000000000000, 600;
			MatrixXd c2(1, 2);
			c2 << 1.73000000000000, 600;
			MatrixXd c3(1, 2);
			c3 << 1.73000000000000, 600;
			vector<MatrixXd> uC = { c1, c2, c3 };

			MatrixXd a1(1, 2);
			a1 << 2, 8;
			MatrixXd a2(1, 2);
			a2 << 4, 4;
			MatrixXd a3(1, 2);
			a3 << 8, 2;
			vector<MatrixXd> uA = { a1, a2, a3 };


			vector<vector<MatrixXd>> four = { uLamb4, uTX, uC, uA };
			return four;
		};

		vector<vector<MatrixXd>> Get5()
		{
			MatrixXd l1(51, 1);
			Load("lamb5_1.txt", l1);
			MatrixXd l2(45, 1);
			Load("lamb5_2.txt", l2);
			MatrixXd l3(45, 1);
			Load("lamb5_3.txt", l3);
			MatrixXd l4(51, 1);
			Load("lamb5_4.txt", l4);
			vector<MatrixXd> uLamb5{ l1,l2,l3, l4 };

			/*MatrixXd tx1(27, 2);
			Load("TX4_1.txt", tx1);
			MatrixXd tx2(25, 2);
			Load("TX4_2.txt", tx2);
			MatrixXd tx3(27, 2);
			Load("TX4_3.txt", tx3);
			vector<MatrixXd> uTX{ tx1,tx2 };

			MatrixXd c1(1, 2);
			c1 << 1.73000000000000, 600;
			MatrixXd c2(1, 2);
			c2 << 1.73000000000000, 600;
			MatrixXd c3(1, 2);
			c3 << 1.73000000000000, 600;
			vector<MatrixXd> uC = { c1, c2, c3 };

			MatrixXd a1(1, 2);
			a1 << 2, 8;
			MatrixXd a2(1, 2);
			a2 << 4, 4;
			MatrixXd a2(1, 2);
			a3 << 8, 2;*/
			//vector<MatrixXd> uA = { a1, a2, a3 };


			vector<vector<MatrixXd>> five = { uLamb5 };// , uTX, uC, uA };
			return five;
		};

		static void Load(string file, MatrixXd &matrix)
		{
			ifstream infile(file);
			vector<double> doubles;
			int count = 0;
			while (infile)
			{
				string s;
				if (!getline(infile, s)) break;

				stringstream ss(s);
				while (ss.good())
				{
					string substr;
					getline(ss, substr, ',');
					doubles.push_back(stod(substr));
				}

				count++;
			}

			int rows = matrix.rows();
			int cols = matrix.cols();
			vector<double>::iterator it;
			int i = 0;
			int rcount = 0;
			int ccount = 0;

			for (it = doubles.begin(); it < doubles.end(); it++, i++) {
				if (ccount > cols - 1)
				{
					ccount = 0;
					rcount++;
				}

				matrix(rcount, ccount) = doubles[i];

				ccount++;

			}

		};

	public:


		TEST_METHOD(TestInterpolate3)
		{
			Interpolation test;
			
			vector<MatrixXd> uLamb3;
			MatrixXd l1(15, 1);
			l1 << 168.177040362907, -384.857133569469, 437.285285931294, -355.212798882940, 443.385219414521, -109.178371522743, 221.964431301908, -185.896683167106, 91.7816691444462, -271.563020925667, 179.474487920222, -412.630669834898, 475.510735359416, -393.047823887079, 457.473155903381;
			uLamb3.push_back(l1);
			MatrixXd l2(15, 1);
			l2 << 266.292960914272, -700.744331062505, 759.616443800679, -184.485437051410, 574.752360107266, -715.267344742912, 324.264851093682, -938.943872118302, 1106.30094053661, -187.682488571149, 578.284311734800, -714.053999346216, 276.596980754002, -715.471078725550, 762.783738957473;
			uLamb3.push_back(l2);

			MatrixXd TX1 = MatrixXd(15, 2);
			TX1 << 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.4325, 0, 0.4325, 75, 0.4325, 150, 0.4325, 225, 0.4325, 300, 0.865, 0, 0.865, 75, 0.865, 150, 0.865, 225, 0.865, 300;
			MatrixXd TX2 = MatrixXd(15, 2);
			TX2 << 0, 0, 0, 150, 0, 300, 0.216250000000000, 0, 0.216250000000000, 150, 0.216250000000000, 300, 0.432500000000000, 0, 0.432500000000000, 150, 0.432500000000000, 300, 0.648750000000000, 0, 0.648750000000000, 150, 0.648750000000000, 300, 0.865000000000000, 0, 0.865000000000000, 150, 0.865000000000000, 300;
			vector<MatrixXd> uTX = { TX1, TX2 };

			MatrixXd c1(1, 2);
			c1 << 1.73000000000000, 600;
			MatrixXd c2(1, 2);
			c2 << 1.73000000000000, 600;
			vector<MatrixXd> uC = { c1, c2 };

			MatrixXd a1(1, 2);
			a1 << 2,4;
			MatrixXd a2(1, 2);
			a2 << 4,2;
			vector<MatrixXd> uA = { a1, a2 };


			map< string, vector<vector<MatrixXd>>> interpolation;
			vector<string> level2 = {};
			test.interpolateGeneric( "3", 2, 0.865, 3, 2, 0, 300, 0.03, 0.15, 1, 100, level2, &interpolation);
			vector<vector<MatrixXd>> result = test.getResult();
			
			vector<MatrixXd> l = result[0];
			for (int i =0 ; i < result[0].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uLamb3[i], l[i], 0.0000000001));
			}

			vector<MatrixXd> tx = result[1];
			for (int i = 0; i < result[1].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uTX[i], tx[i]));
			}

			vector<MatrixXd> c = result[2];
			for (int i = 0; i < result[2].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uC[i], c[i]));
			}

			vector<MatrixXd> a = result[3];
			for (int i = 0; i < result[3].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uA[i], a[i]));
			}
			
			//Assert::IsTrue(testCommon::checkMatrix(expected, result));
		}

		TEST_METHOD(TestInterpolate_3)
		{
			Interpolation test;

			vector<vector<MatrixXd>> _3 = Get_3();

			map< string, vector<vector<MatrixXd>>> interpolation;
			
			interpolation["3"] = Get3();
			interpolation["2"] = Get2();

			vector<string> level3 = { "2","3" };
			test.interpolateGeneric("_3", 2, 0.865, 3, 2, 0, 300, 0.03, 0.15, 1, 100, level3, &interpolation);
			vector<vector<MatrixXd>> result = test.getResult();

			auto uLamb3 = _3[0];
			vector<MatrixXd> l = result[0];
			for (int i = 0; i < result[0].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uLamb3[i], l[i], 0.0000000001));
			}

			auto uTX = _3[1];
			vector<MatrixXd> tx = result[1];
			for (int i = 0; i < result[1].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uTX[i], tx[i]));
			}

			auto uC = _3[2];
			vector<MatrixXd> c = result[2];
			for (int i = 0; i < result[2].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uC[i], c[i]));
			}

			auto uA = _3[3];
			vector<MatrixXd> a = result[3];
			for (int i = 0; i < result[3].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uA[i], a[i]));
			}

		}

		TEST_METHOD(TestInterpolate5)
		{
			Interpolation test;

			map< string, vector<vector<MatrixXd>>> interpolation;

			interpolation["2"] = Get2();
			interpolation["3"] = Get3();
			interpolation["_3"] = Get_3();
			interpolation["4"] = Get4();

			interpolation["5"] = Get5();

			vector<string> level4 = { "2","3", "_3","4" };
			test.interpolateGeneric("5", 2, 0.865, 5, 2, 0, 300, 0.03, 0.15, 1, 100, level4, &interpolation);
			vector<vector<MatrixXd>> result = test.getResult();

			auto uLamb5 = interpolation["5"][0];
			vector<MatrixXd> l = result[0];
			for (int i = 0; i < result[0].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uLamb5[i], l[i], 0.000000001));
			}

			/*auto uTX = _3[1];
			vector<MatrixXd> tx = result[1];
			for (int i = 0; i < result[1].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uTX[i], tx[i]));
			}

			auto uC = _3[2];
			vector<MatrixXd> c = result[2];
			for (int i = 0; i < result[2].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uC[i], c[i]));
			}

			auto uA = _3[3];
			vector<MatrixXd> a = result[3];
			for (int i = 0; i < result[3].size(); i++)
			{
				Assert::IsTrue(testCommon::checkMatrix(uA[i], a[i]));
			}*/

		}

		TEST_METHOD(TestSubnumber)
		{
			Interpolation test;

			MatrixXd expected(2, 2);
			expected << 1, 2, 2, 1;
			MatrixXd result = test.subnumber(3,2);

			//Assert::IsTrue(testCommon::checkMatrix(expected, result));

		}

		TEST_METHOD(TestInnerTest)
		{
			double d = 1.681770403629075 * 100;

			Test test;

			double expected = -2.2427e-13;
			

			MatrixXd l1, l2;
			//l1 = MatrixXd(15, 1);
			//l2 = MatrixXd(15, 1);
			//l1 << 168.1770403629075, -384.8571335694688, 437.2852859312935, -355.2127988829399, 443.3852194145209, -109.1783715227434, 221.9644313019083, -185.8966831671062, 91.7816691444462, -271.5630209256668, 179.4744879202224, -412.6306698348984, 475.5107353594159, -393.0478238870790, 457.4731559033815;
			//l2 << 0.266292960914272, -0.700744331062505, 0.759616443800679, -0.184485437051410, 0.574752360107266, -0.715267344742912, 0.324264851093682, -0.938943872118302, 1.106300940536614, -0.187682488571149, 0.578284311734800, -0.714053999346216, 0.276596980754002, -0.715471078725550, 0.762783738957473;
			l1 = LoadData("inner_testLamb3_1.txt", 15, 15, 1);
			Logger::WriteMessage(Common::printMatrix(l1).c_str());
			l2 = LoadData("inner_testLamb3_2.txt", 15, 15, 1);
			Logger::WriteMessage(Common::printMatrix(l2).c_str());
			vector<MatrixXd> lambda3 = { l1.array(),l2.array()};

			MatrixXd tx1, tx2;
			tx1 = MatrixXd(15, 2);
			tx2 = MatrixXd(15, 2);
			tx1 << 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.432500000000000, 0, 0.432500000000000, 75, 0.432500000000000, 150, 0.432500000000000, 225, 0.432500000000000, 300, 0.865000000000000, 0, 0.865000000000000, 75, 0.865000000000000, 150, 0.865000000000000, 225, 0.865000000000000, 300;
			tx2 << 0, 0, 0, 150, 0, 300, 0.216250000000000, 0, 0.216250000000000, 150, 0.216250000000000, 300, 0.432500000000000, 0, 0.432500000000000, 150, 0.432500000000000, 300, 0.648750000000000, 0, 0.648750000000000, 150, 0.648750000000000, 300, 0.865000000000000, 0, 0.865000000000000, 150, 0.865000000000000, 300;
			vector<MatrixXd> TX3 = { tx1,tx2 };

			MatrixXd c1, c2;
			c1 = MatrixXd(1, 2);
			c2 = MatrixXd(1, 2);
			c1 << 1.73000000000000,	600;
			c2 << 1.73000000000000,	600;
			vector<MatrixXd> C3 = { c1,c2 };

			MatrixXd a1, a2;
			a1 = MatrixXd(1, 2);
			a2 = MatrixXd(1, 2);
			a1 << 2, 4;
			a2 << 4, 2;
			vector<MatrixXd> A3 = { a1,a2 };

			double result = test.innerZ(0, 0,  lambda3, TX3, C3, A3);
			//char buff[256];
			//sprintf(buff, "%f", result);
			string s = std::to_string(result);
			Logger::WriteMessage( s.c_str() );
			Assert::AreEqual(expected, result, 0.0000000000001);
			//Assert::AreEqual(expected, result);

		}

		TEST_METHOD(TestInnerMock)
		{
			
			Test test;

			double result = test.innerMock("10", 0, 0, 0);

			result = test.innerMock("_9", 7, 0, 1);

			result = test.innerMock("_10", 0, 0, 1);
			
		}

		TEST_METHOD(TestVectorMultiplication)
		{
			//MatrixXd a(1, 15);
			//a << 1.000000000000000,   0.778800783071405,   0.367879441171442,   0.105399224561864,   0.018315638888734,   0.778800783071405,   0.606530659712633,   0.286504796860190, 0.082084998623899,   0.014264233908999,   0.367879441171442,   0.286504796860190,   0.135335283236613,   0.038774207831722,   0.006737946999085;
			//a << 1.0000, 0.7788, 0.3679, 0.1054, 0.0183, 0.7788, 0.6065, 0.2865, 0.0821, 0.0143, 0.3679, 0.2865, 0.1353, 0.0388, 0.0067;
			//a << 1.0000, 0.7788, 0.3679, 0.1054, 0.0183, 0.7788, 0.6065, 0.2865, 0.0821, 0.0143;
			MatrixXd a = LoadData("inner_testD.txt", 15, 1, 15);
			//MatrixXd b(15,1);
			//b << 1.681770403629075, -3.848571335694688, 4.372852859312935, -3.552127988829399, 4.433852194145209, -1.091783715227434, 2.219644313019083, -1.858966831671062, 0.917816691444462, -2.715630209256668, 1.794744879202224, -4.126306698348984, 4.755107353594159, -3.930478238870790, 4.574731559033815;
			//b = b.array() * 100;
			//b << 168.1770, -384.8571, 437.2853, -355.2128, 443.3852, -109.1784, 221.9644, -185.8967, 91.7817, -271.5630, 179.4745, -412.6307, 475.5107, -393.0478, 457.4732;
			//b << 168.1770, -384.8571, 437.2853, -355.2128, 443.3852, -109.1784, 221.9644, -185.8967, 91.7817, -271.5630;
			MatrixXd b = LoadData("inner_testLamb.txt", 15, 15, 1);
			//MatrixXdM c = MatrixXdM(a) * MatrixXdM(b);
			MatrixXdM c = MatrixXdM(a) * MatrixXdM(b);
			Logger::WriteMessage(Common::printMatrix(a).c_str());
			Logger::WriteMessage(Common::printMatrix(b).c_str());
			Logger::WriteMessage(Common::printMatrix(c.value()).c_str());

			Assert::AreEqual(-3.0642e-14, c.value()(0, 0), 0.000000000001);

		}

		typedef number<cpp_dec_float<14> > cpp_dec_float_14;
		typedef number<cpp_dec_float<15> > cpp_dec_float_15;
		typedef number<cpp_dec_float<16> > cpp_dec_float_16;
		typedef number<cpp_dec_float<17> > cpp_dec_float_17;
		typedef number<cpp_dec_float<18> > cpp_dec_float_18;
		typedef number<cpp_dec_float<19> > cpp_dec_float_19;
		typedef number<cpp_dec_float<20> > cpp_dec_float_20;

		//TEST_METHOD(TestVectorMultiplication1)
		//{


		//	MatrixXd a = Common::ReadBinary("d_transpose.dat", 1, 15);
		//	
		//	MatrixXd b = Common::ReadBinary("lambj.dat", 15, 1);
		//	
		//	MatrixXd c = a * b;
		//	MatrixXdM c1 = MatrixXdM(a) * MatrixXdM(b);
		//	vector<cpp_dec_float_14> c2 = mult(MatrixXdM(a), MatrixXdM(b));

		//	Logger::WriteMessage(Common::printMatrix(a).c_str());
		//	Logger::WriteMessage(Common::printMatrix(b).c_str());
		//	Logger::WriteMessage(Common::printMatrix(c).c_str());
		//	Logger::WriteMessage(Common::printMatrix(c1.value()).c_str());
		//	
		//	for (auto i : c2)
		//	{
		//		Logger::WriteMessage(i.str().c_str());

		//	}

		//	Assert::AreEqual(-3.0642e-14, c1.value()(0, 0), 0.000000000001);

		//}

		vector<cpp_dec_float_14> mult(MatrixXdM &a, MatrixXdM &b)
		{
			//MatrixXd result(a.value().rows(), b.value().cols());
			vector<cpp_dec_float_14> result;
			//assume a & b are compatible
			for (int i = 0; i < a.value().rows(); i++)
				for (int j = 0; j < b.value().cols(); j++)
				{
					cpp_dec_float_14 sum = 0;
					for (int x = 0; x < a.value().cols(); x++)
					{
						cpp_dec_float_14 l = a.value()(i, x);
						cpp_dec_float_14 r = b.value()(x, j);
						sum = sum + (l * r) ;

						result.push_back(sum);
					}
					
				}
			return result;

		};

		MatrixXd LoadData(string fileName, int size, int rows, int columns)
		{
			const int N = 15;

			double *U = (double*)malloc(N * sizeof(double));

			std::ifstream infile;
			//infile.open(L"C:\\Users\\User\\Documents\\Visual Studio 2017\\Projects\\ConsoleApplication1\\x64\\Debug\\inner_testD.txt", std::ios::in | std::ios::binary);
			string path = "C:\\Users\\User\\Source\\Repos\\SparseGridCollocation\\SparseGridCollocation\\x64\\Debug\\";
			string fpath = path + fileName;
			infile.open(fpath, std::ios::in | std::ios::binary);
			infile.read((char*)U, size * sizeof(double));
			infile.close();

			MatrixXd result = MatrixXd::Zero(rows, columns);
			int rowCount = 0, colCount = 0;
			for (int i = 0; i < size; i++)
			{
				result(rowCount, colCount) = U[i];
				if (colCount == columns - 1)
				{
					colCount = 0;
					rowCount++;
				}
				else
					colCount++;
			}

			delete U;

			Logger::WriteMessage(Common::printMatrix(result).c_str());
			return result;
		};



	};
}