// TestHarness.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
//#include "CppUnitTest.h"
//#include "SparseGridCollocation.h"
//#include <Eigen/Dense>
#include "Math.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>

#include "Common.h"
using namespace Eigen;
using namespace std;




class data {

public:
	static MatrixXd GetTX()
	{
		MatrixXd TX(10000, 2);
		Load("TX.txt", TX);
		//cout << Common::printMatrixA(TX) << endl;
		return TX;
	};

	//static void GetLamb_10(vector<MatrixXd> &result)
	//{
	//	/*unique_ptr<MatrixXd> lamb1( new MatrixXd (1536,1));
	//	Load("lamb1.txt", move(lamb1));*/
	//	MatrixXd lamb1( 1536,1);
	//	Load("lamb1.txt", lamb1);
	//	MatrixXd lamb2(1285, 1);
	//	Load("lamb2.txt", lamb2);
	//	MatrixXd lamb3(1161, 1);
	//	Load("lamb3.txt", lamb3);
	//	MatrixXd lamb4(1105, 1);
	//	Load("lamb4.txt", lamb4);
	//	MatrixXd lamb5(1089, 1);
	//	Load("lamb5.txt", lamb5);
	//	MatrixXd lamb6(1105, 1);
	//	Load("lamb6.txt", lamb6);
	//	MatrixXd lamb7(1161, 1);
	//	Load("lamb7.txt", lamb7);
	//	MatrixXd lamb8(1285, 1);
	//	Load("lamb8.txt", lamb8);
	//	MatrixXd lamb9(1539, 1);
	//	Load("lamb9.txt", lamb9);

	//	//vector<MatrixXd*> Lamb10 = { &lamb1,&lamb2,&lamb3 ,&lamb4 ,&lamb5 ,&lamb6 ,&lamb7 ,&lamb8 ,&lamb9 };
	//	//new <vector<MatrixXd>>{ &lamb1,&lamb2,&lamb3 ,&lamb4 ,&lamb5 ,&lamb6 ,&lamb7 ,&lamb8 ,&lamb9 }
	//	//unique_ptr<vector<MatrixXd>> Lamb10(<vector<MatrixXd>>());
	//	//result = { lamb1, lamb2, lamb3 , lamb4 ,lamb5 ,lamb6 ,lamb7 ,lamb8 ,lamb9 };
	//	result = { lamb1, lamb2, lamb3 , lamb4 ,lamb5 ,lamb6 ,lamb7 ,lamb8 ,lamb9 };
	//	//return Lamb10;
	//};

	static MatrixXd GetLamb2()
	{
		MatrixXd result(9, 1);
		result << 273.169921181606, -696.599817216584, 732.148991790275, -191.889255272131, 521.665751945236, -582.757329851410, 284.102691920217, -712.786022564190, 736.696836746533;
		return result;
	}

	static MatrixXd GetLamb3(int n)
	{
		MatrixXd result(15, 1);
		if (n == 1)
		{
			result << 168.177040362907, -384.857133569469, 437.285285931294, -355.212798882940, 443.385219414521, -109.178371522743, 221.964431301908, -185.896683167106, 91.7816691444462, -271.563020925667, 179.474487920222, -412.630669834898, 475.510735359416, -393.047823887079, 457.473155903381;
		}
		if (n == 2)
		{
			result << 266.292960914272, -700.744331062505, 759.616443800679, -184.485437051410, 574.752360107266, -715.267344742912, 324.264851093682, -938.943872118302, 1106.30094053661, -187.682488571149, 578.284311734800, -714.053999346216, 276.596980754002, -715.471078725550, 762.783738957473;
		}
		return result;
	}

	static MatrixXd GetLamb_3(int n)
	{
		MatrixXd result(15, 1);
		if (n == 1)
		{
			result << -42.8498877775422, 141.016331069450, -249.998444271759, 258.615502944090, -123.518857109165, 65.0489756228674, -219.762990596407, 397.963768108779, -417.194722894546, 200.480323513864, -34.8966003832208, 119.274580096719, -217.965206264739, 229.772250182186, -110.694184815550;
		}
		if (n == 2)
		{
			result << -0.0568069765069908, 0.0997753173328967, -0.0568069764883387, 0.0369586729170275, -0.0649139162828327, 0.0369586728689974, -0.0138051312405887, 0.0242472216107991, -0.0138051311798996, -0.0763568261614475, 0.134112516310045, -0.0763568262085483, 0.0616904205757057, -0.108352559302475, 0.0616904205955072;
		}
		return result;
	}

	static MatrixXd GetLamb4(int n)
	{
		MatrixXd result(15, 1);
		if (n == 1)
		{
			result << -45.7851033398151, 94.1526208090134, -89.3228152872666, 67.6613254864874, -120.633852215053, 224.944131760658, -313.852910653178, 321.102166398442, -156.234075391722, 24.8104612277631, -52.6539041920979, 53.4415775491540, -53.3818114478420, 147.673985193500, -308.726792415997, 456.713692484328, -494.615449397829, 247.122847566978, 14.0405791777674, -89.1444335651629, 241.130983185538, -374.898202880229, 361.985086984658, -231.175356792814, 64.5448233225828, 73.3246087342294, -62.4198701820482;
		}
		if (n == 2)
		{
			result << -45.1157821994960, 156.391073363218, -288.503108078103, 304.992695560773, -147.051361443715, 113.909367634626, -395.680506999585, 731.042880180116, -773.452953911068, 373.048215710146, -144.853114728980, 508.215899837437, -945.862128867320, 1004.84710087542, -485.524962047430, 109.412218092133, -383.604142807472, 713.760684406116, -758.582815786136, 366.634633763299, -43.1014728733641, 150.629041768111, -279.681382494740, 297.057506527674, -143.546341188895;
		}
		if (n == 3)
		{
			result << 21.9174411350081, -51.1754163784310, 48.7605753596712, -41.6021405981585, 103.524473604306, -106.074969527104, 32.4682018149328, -97.2468698544000, 117.613902100889, -27.1157406958468, 89.8658588043425, -116.537799382749, 23.5907678448124, -83.5002162473479, 112.643603630529, -27.4986528043536, 90.1508453720181, -116.100252924641, 33.3220593267526, -97.9386455379889, 116.757365875791, -43.7594556071896, 106.421281744648, -106.343322220755, 24.3361898309736, -54.9858030758848, 50.2523234335064;
		}
		return result;
	}

	static MatrixXd GetLamb_10(int n)
	{
		/*unique_ptr<MatrixXd> lamb1( new MatrixXd (1536,1));
		Load("lamb1.txt", move(lamb1));*/
		MatrixXd result;
		if (n == 1)
		{
			result = MatrixXd(1539, 1);
			Load("lamb1.txt", result);
		}
		if (n == 2)
		{
			result = MatrixXd(1285, 1);
			Load("lamb2.txt", result);
		}
		if (n == 3)
		{
			result = MatrixXd(1161, 1);
			Load("lamb3.txt", result);
		}
		if (n == 4)
		{
			result = MatrixXd(1105, 1);
			Load("lamb4.txt", result);
		}
		if (n == 5)
		{
			result = MatrixXd(1089, 1);
			Load("lamb5.txt", result);
		}
		if (n == 6)
		{
			result = MatrixXd(1105, 1);
			Load("lamb6.txt", result);
		}
		if (n == 7)
		{
			result = MatrixXd(1161, 1);
			Load("lamb7.txt", result);
		}
		if (n == 8)
		{
			result = MatrixXd(1285, 1);
			Load("lamb8.txt", result);
		}
		if (n == 9)
		{
			result = MatrixXd(1539, 1);
			Load("lamb9.txt", result);
		}
		return result;
	};

	//static vector<MatrixXd> GetTX_10()
	//{

	//	MatrixXd TX1(1539, 2);
	//	TX1 = Load("TX1.txt");
	//	MatrixXd TX2(1539, 2);
	//	TX2 = Load("TX2.txt");
	//	MatrixXd TX3(1539, 2);
	//	TX3 = Load("TX3.txt");
	//	MatrixXd TX4(1539, 2);
	//	TX4 = Load("TX4.txt");
	//	MatrixXd TX5(1539, 2);
	//	TX5 = Load("TX5.txt");
	//	MatrixXd TX6(1539, 2);
	//	TX6 = Load("TX6.txt");
	//	MatrixXd TX7(1539, 2);
	//	TX7 = Load("TX7.txt");
	//	MatrixXd TX8(1539, 2);
	//	TX8 = Load("TX8.txt");
	//	MatrixXd TX9(1539, 2);
	//	TX9 = Load("TX9.txt");

	//	vector<MatrixXd> TX_10 = { TX1,TX2, TX3, TX4, TX5, TX6, TX7, TX8, TX9 };
	//	return TX_10;
	//};

	static vector<MatrixXd> GetC_10()
	{
		MatrixXd C1(1, 2);
		C1 << 1.73, 600;
		MatrixXd C2(1, 2);
		C2 << 1.73, 600;
		MatrixXd C3(1, 2);
		C3 << 1.73, 600;
		MatrixXd C4(1, 2);
		C4 << 1.73, 600;
		MatrixXd C5(1, 2);
		C5 << 1.73, 600;
		MatrixXd C6(1, 2);
		C6 << 1.73, 600;
		MatrixXd C7(1, 2);
		C7 << 1.73, 600;
		MatrixXd C8(1, 2);
		C8 << 1.73, 600;
		MatrixXd C9(1, 2);
		C9 << 1.73, 600;

		vector<MatrixXd> C_10 = { C1,C2, C3, C4, C5, C6, C7, C8, C9 };
		return C_10;
	};

	static vector<MatrixXd> GetA_10()
	{
		MatrixXd A1(1, 2);
		A1 << 2, 512;
		MatrixXd A2(1, 2);
		A2 << 4, 256;
		MatrixXd A3(1, 2);
		A3 << 8, 128;
		MatrixXd A4(1, 2);
		A4 << 16, 64;
		MatrixXd A5(1, 2);
		A5 << 32, 32;
		MatrixXd A6(1, 2);
		A6 << 64, 16;
		MatrixXd A7(1, 2);
		A7 << 128, 8;
		MatrixXd A8(1, 2);
		A8 << 256, 4;
		MatrixXd A9(1, 2);
		A9 << 512, 2;

		vector<MatrixXd> A_10 = { A1,A2, A3, A4, A5, A6, A7, A8, A9 };
		return A_10;
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
		
	}
};


int main()
{

	SparseGridCollocation* test = new SparseGridCollocation();
	vector<VectorXd> result = test->MuSIKGeneric(11);
	wcout << Common::printMatrix(result[0]) << endl;
	wcout << Common::printMatrix(result[1]) << endl;

	//uncomment below to test each interpolation level:
	//MatrixXd l=data::GetLamb_2();

	//cout << Common::printMatrixA(l) << endl;
	
	//VectorXd uRMS(9);
	//uRMS << 3.74273075820591, 1.39067846770261, 0.436408235898941, 0.122815017267700, 0.0328347672816514, 0.00835749999263466, 0.00221774306858896, 0.000597422702846034, 0.000162823604659233;
	//VectorXd uMax = VectorXd(9);
	//uMax << 6.55956979634471, 3.70216720531096, 1.42734558657324, 0.416600021955134, 0.114733914477483, 0.0293262625138198, 0.00765938580167358, 0.00198337219728728, 0.000513094949184278;

	//MatrixXd t = data::GetTX();
	//Common::printMatrixA(t);

	//vector<VectorXd> result = test->MuSIKGeneric(3);
	//map<string, vector<vector<MatrixXd>>> interpolation = test->GetInterpolation();

	//vector<vector<MatrixXd>>::iterator it;
	//vector<vector<MatrixXd>> _10 = interpolation["_10"];
	//vector<vector<MatrixXd>> item2 = interpolation["2"];
	//vector<vector<MatrixXd>> item3 = interpolation["3"];
	//vector<vector<MatrixXd>> item_3 = interpolation["_3"];
	//vector<vector<MatrixXd>> item4 = interpolation["4"];
	//int count = 0;
	////for (it = _10.begin(); it < _10.end(); it++, count++)
	//for (it = item_3.begin(); it < item_3.end(); it++, count++)
	//{
	//	vector<MatrixXd> vExpected;
	//	//if (count == 0)
	//	//{
	//	//	MatrixXd actual = _10[count][0];
	//	//	MatrixXd expected = data::GetTX();
	//	//	if (!Common::checkMatrix(expected, actual, 0.0000001))
	//	//		cout << "check TX failed" << endl;
	//	//}
	//	//if (count == 0)
	//	//	data::GetLamb_10(vExpected);
	//	////if (count == 2)
	//	////	vExpected = data::GetTX_10();
	//	//if (count == 2)
	//	//	vExpected = data::GetC_10();
	//	//if (count == 3)
	//	//	vExpected = data::GetA_10();

	//	//vector<MatrixXd> item = item_3[count];
	//	vector<MatrixXd> item = item4[count];
	//	//vector<MatrixXd> item = _10[count];
	//	vector<MatrixXd>::iterator it2;
	//	int count2 = 0;
	//	for (it2 = item.begin(); it2 < item.end(); it2++, count2++)
	//	{

	//		MatrixXd actual = item[count2];
	//		//MatrixXd expected = vExpected[count2];
	//		//MatrixXd expected = data::GetLamb_10(count2 +1);
	//		//MatrixXd expected = data::GetLamb_3(count2 + 1);
	//		MatrixXd expected = data::GetLamb4(count2 + 1);
	//		if (!Common::checkMatrix(expected, actual, 0.6))
	//		{
	//			if (count == 1)
	//				cout << "check Lamb_10 failed" << endl;
	//			if (count == 2)
	//				cout << "check TX_10 failed" << endl;
	//			if (count == 3)
	//				cout << "check C_10 failed" << endl;
	//			if (count == 4)
	//				cout << "check A_10 failed" << endl;
	//		}
	//	}
	//}

	/*interpolation["11"];

	wstring msg = Common::printMatrix(result[0]);
	wcout << msg << endl;

	if (!Common::checkMatrix(uRMS, result[0], 0.0000001))
		cout << "check RMS failed" << endl;


	msg = Common::printMatrix(result[1]);
	wcout << msg << endl;


	if (!Common::checkMatrix(uMax, result[1], 0.0000001))
		cout << "check MAX failed" << endl;*/

	return 0;
}

