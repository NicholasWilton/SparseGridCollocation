#include "Gaussian2d.h"
#include "Gaussian2d2.h"
#include "GaussianNd1.h"
#include "Common.h"
#include "Utility.h"
#include "NodeRegistry.h"
#include "cuda_runtime.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;
using namespace thrust; 
using namespace Leicester::ThrustLib;


MatrixXd GetTX7()
{
	MatrixXd TX1(195, 2);
	TX1(0, 0) = 0;
	TX1(1, 0) = 0;
	TX1(2, 0) = 0;
	TX1(3, 0) = 0;
	TX1(4, 0) = 0;
	TX1(5, 0) = 0;
	TX1(6, 0) = 0;
	TX1(7, 0) = 0;
	TX1(8, 0) = 0;
	TX1(9, 0) = 0;
	TX1(10, 0) = 0;
	TX1(11, 0) = 0;
	TX1(12, 0) = 0;
	TX1(13, 0) = 0;
	TX1(14, 0) = 0;
	TX1(15, 0) = 0;
	TX1(16, 0) = 0;
	TX1(17, 0) = 0;
	TX1(18, 0) = 0;
	TX1(19, 0) = 0;
	TX1(20, 0) = 0;
	TX1(21, 0) = 0;
	TX1(22, 0) = 0;
	TX1(23, 0) = 0;
	TX1(24, 0) = 0;
	TX1(25, 0) = 0;
	TX1(26, 0) = 0;
	TX1(27, 0) = 0;
	TX1(28, 0) = 0;
	TX1(29, 0) = 0;
	TX1(30, 0) = 0;
	TX1(31, 0) = 0;
	TX1(32, 0) = 0;
	TX1(33, 0) = 0;
	TX1(34, 0) = 0;
	TX1(35, 0) = 0;
	TX1(36, 0) = 0;
	TX1(37, 0) = 0;
	TX1(38, 0) = 0;
	TX1(39, 0) = 0;
	TX1(40, 0) = 0;
	TX1(41, 0) = 0;
	TX1(42, 0) = 0;
	TX1(43, 0) = 0;
	TX1(44, 0) = 0;
	TX1(45, 0) = 0;
	TX1(46, 0) = 0;
	TX1(47, 0) = 0;
	TX1(48, 0) = 0;
	TX1(49, 0) = 0;
	TX1(50, 0) = 0;
	TX1(51, 0) = 0;
	TX1(52, 0) = 0;
	TX1(53, 0) = 0;
	TX1(54, 0) = 0;
	TX1(55, 0) = 0;
	TX1(56, 0) = 0;
	TX1(57, 0) = 0;
	TX1(58, 0) = 0;
	TX1(59, 0) = 0;
	TX1(60, 0) = 0;
	TX1(61, 0) = 0;
	TX1(62, 0) = 0;
	TX1(63, 0) = 0;
	TX1(64, 0) = 0;
	TX1(65, 0) = 0.432499999999999;
	TX1(66, 0) = 0.432499999999999;
	TX1(67, 0) = 0.432499999999999;
	TX1(68, 0) = 0.432499999999999;
	TX1(69, 0) = 0.432499999999999;
	TX1(70, 0) = 0.432499999999999;
	TX1(71, 0) = 0.432499999999999;
	TX1(72, 0) = 0.432499999999999;
	TX1(73, 0) = 0.432499999999999;
	TX1(74, 0) = 0.432499999999999;
	TX1(75, 0) = 0.432499999999999;
	TX1(76, 0) = 0.432499999999999;
	TX1(77, 0) = 0.432499999999999;
	TX1(78, 0) = 0.432499999999999;
	TX1(79, 0) = 0.432499999999999;
	TX1(80, 0) = 0.432499999999999;
	TX1(81, 0) = 0.432499999999999;
	TX1(82, 0) = 0.432499999999999;
	TX1(83, 0) = 0.432499999999999;
	TX1(84, 0) = 0.432499999999999;
	TX1(85, 0) = 0.432499999999999;
	TX1(86, 0) = 0.432499999999999;
	TX1(87, 0) = 0.432499999999999;
	TX1(88, 0) = 0.432499999999999;
	TX1(89, 0) = 0.432499999999999;
	TX1(90, 0) = 0.432499999999999;
	TX1(91, 0) = 0.432499999999999;
	TX1(92, 0) = 0.432499999999999;
	TX1(93, 0) = 0.432499999999999;
	TX1(94, 0) = 0.432499999999999;
	TX1(95, 0) = 0.432499999999999;
	TX1(96, 0) = 0.432499999999999;
	TX1(97, 0) = 0.432499999999999;
	TX1(98, 0) = 0.432499999999999;
	TX1(99, 0) = 0.432499999999999;
	TX1(100, 0) = 0.432499999999999;
	TX1(101, 0) = 0.432499999999999;
	TX1(102, 0) = 0.432499999999999;
	TX1(103, 0) = 0.432499999999999;
	TX1(104, 0) = 0.432499999999999;
	TX1(105, 0) = 0.432499999999999;
	TX1(106, 0) = 0.432499999999999;
	TX1(107, 0) = 0.432499999999999;
	TX1(108, 0) = 0.432499999999999;
	TX1(109, 0) = 0.432499999999999;
	TX1(110, 0) = 0.432499999999999;
	TX1(111, 0) = 0.432499999999999;
	TX1(112, 0) = 0.432499999999999;
	TX1(113, 0) = 0.432499999999999;
	TX1(114, 0) = 0.432499999999999;
	TX1(115, 0) = 0.432499999999999;
	TX1(116, 0) = 0.432499999999999;
	TX1(117, 0) = 0.432499999999999;
	TX1(118, 0) = 0.432499999999999;
	TX1(119, 0) = 0.432499999999999;
	TX1(120, 0) = 0.432499999999999;
	TX1(121, 0) = 0.432499999999999;
	TX1(122, 0) = 0.432499999999999;
	TX1(123, 0) = 0.432499999999999;
	TX1(124, 0) = 0.432499999999999;
	TX1(125, 0) = 0.432499999999999;
	TX1(126, 0) = 0.432499999999999;
	TX1(127, 0) = 0.432499999999999;
	TX1(128, 0) = 0.432499999999999;
	TX1(129, 0) = 0.432499999999999;
	TX1(130, 0) = 0.864999999999999;
	TX1(131, 0) = 0.864999999999999;
	TX1(132, 0) = 0.864999999999999;
	TX1(133, 0) = 0.864999999999999;
	TX1(134, 0) = 0.864999999999999;
	TX1(135, 0) = 0.864999999999999;
	TX1(136, 0) = 0.864999999999999;
	TX1(137, 0) = 0.864999999999999;
	TX1(138, 0) = 0.864999999999999;
	TX1(139, 0) = 0.864999999999999;
	TX1(140, 0) = 0.864999999999999;
	TX1(141, 0) = 0.864999999999999;
	TX1(142, 0) = 0.864999999999999;
	TX1(143, 0) = 0.864999999999999;
	TX1(144, 0) = 0.864999999999999;
	TX1(145, 0) = 0.864999999999999;
	TX1(146, 0) = 0.864999999999999;
	TX1(147, 0) = 0.864999999999999;
	TX1(148, 0) = 0.864999999999999;
	TX1(149, 0) = 0.864999999999999;
	TX1(150, 0) = 0.864999999999999;
	TX1(151, 0) = 0.864999999999999;
	TX1(152, 0) = 0.864999999999999;
	TX1(153, 0) = 0.864999999999999;
	TX1(154, 0) = 0.864999999999999;
	TX1(155, 0) = 0.864999999999999;
	TX1(156, 0) = 0.864999999999999;
	TX1(157, 0) = 0.864999999999999;
	TX1(158, 0) = 0.864999999999999;
	TX1(159, 0) = 0.864999999999999;
	TX1(160, 0) = 0.864999999999999;
	TX1(161, 0) = 0.864999999999999;
	TX1(162, 0) = 0.864999999999999;
	TX1(163, 0) = 0.864999999999999;
	TX1(164, 0) = 0.864999999999999;
	TX1(165, 0) = 0.864999999999999;
	TX1(166, 0) = 0.864999999999999;
	TX1(167, 0) = 0.864999999999999;
	TX1(168, 0) = 0.864999999999999;
	TX1(169, 0) = 0.864999999999999;
	TX1(170, 0) = 0.864999999999999;
	TX1(171, 0) = 0.864999999999999;
	TX1(172, 0) = 0.864999999999999;
	TX1(173, 0) = 0.864999999999999;
	TX1(174, 0) = 0.864999999999999;
	TX1(175, 0) = 0.864999999999999;
	TX1(176, 0) = 0.864999999999999;
	TX1(177, 0) = 0.864999999999999;
	TX1(178, 0) = 0.864999999999999;
	TX1(179, 0) = 0.864999999999999;
	TX1(180, 0) = 0.864999999999999;
	TX1(181, 0) = 0.864999999999999;
	TX1(182, 0) = 0.864999999999999;
	TX1(183, 0) = 0.864999999999999;
	TX1(184, 0) = 0.864999999999999;
	TX1(185, 0) = 0.864999999999999;
	TX1(186, 0) = 0.864999999999999;
	TX1(187, 0) = 0.864999999999999;
	TX1(188, 0) = 0.864999999999999;
	TX1(189, 0) = 0.864999999999999;
	TX1(190, 0) = 0.864999999999999;
	TX1(191, 0) = 0.864999999999999;
	TX1(192, 0) = 0.864999999999999;
	TX1(193, 0) = 0.864999999999999;
	TX1(194, 0) = 0.864999999999999;
	TX1(0, 1) = 0;
	TX1(1, 1) = 4.6875;
	TX1(2, 1) = 9.375;
	TX1(3, 1) = 14.0625;
	TX1(4, 1) = 18.75;
	TX1(5, 1) = 23.4375;
	TX1(6, 1) = 28.125;
	TX1(7, 1) = 32.8125;
	TX1(8, 1) = 37.5;
	TX1(9, 1) = 42.1875;
	TX1(10, 1) = 46.875;
	TX1(11, 1) = 51.5625;
	TX1(12, 1) = 56.25;
	TX1(13, 1) = 60.9375;
	TX1(14, 1) = 65.625;
	TX1(15, 1) = 70.3125;
	TX1(16, 1) = 75;
	TX1(17, 1) = 79.6875;
	TX1(18, 1) = 84.375;
	TX1(19, 1) = 89.0625;
	TX1(20, 1) = 93.75;
	TX1(21, 1) = 98.4375;
	TX1(22, 1) = 103.125;
	TX1(23, 1) = 107.8125;
	TX1(24, 1) = 112.5;
	TX1(25, 1) = 117.1875;
	TX1(26, 1) = 121.875;
	TX1(27, 1) = 126.5625;
	TX1(28, 1) = 131.25;
	TX1(29, 1) = 135.9375;
	TX1(30, 1) = 140.625;
	TX1(31, 1) = 145.3125;
	TX1(32, 1) = 150;
	TX1(33, 1) = 154.6875;
	TX1(34, 1) = 159.375;
	TX1(35, 1) = 164.0625;
	TX1(36, 1) = 168.75;
	TX1(37, 1) = 173.4375;
	TX1(38, 1) = 178.125;
	TX1(39, 1) = 182.8125;
	TX1(40, 1) = 187.5;
	TX1(41, 1) = 192.1875;
	TX1(42, 1) = 196.875;
	TX1(43, 1) = 201.5625;
	TX1(44, 1) = 206.25;
	TX1(45, 1) = 210.9375;
	TX1(46, 1) = 215.625;
	TX1(47, 1) = 220.3125;
	TX1(48, 1) = 225;
	TX1(49, 1) = 229.6875;
	TX1(50, 1) = 234.375;
	TX1(51, 1) = 239.0625;
	TX1(52, 1) = 243.75;
	TX1(53, 1) = 248.4375;
	TX1(54, 1) = 253.125;
	TX1(55, 1) = 257.8125;
	TX1(56, 1) = 262.5;
	TX1(57, 1) = 267.1875;
	TX1(58, 1) = 271.875;
	TX1(59, 1) = 276.5625;
	TX1(60, 1) = 281.25;
	TX1(61, 1) = 285.9375;
	TX1(62, 1) = 290.625;
	TX1(63, 1) = 295.3125;
	TX1(64, 1) = 300;
	TX1(65, 1) = 0;
	TX1(66, 1) = 4.6875;
	TX1(67, 1) = 9.375;
	TX1(68, 1) = 14.0625;
	TX1(69, 1) = 18.75;
	TX1(70, 1) = 23.4375;
	TX1(71, 1) = 28.125;
	TX1(72, 1) = 32.8125;
	TX1(73, 1) = 37.5;
	TX1(74, 1) = 42.1875;
	TX1(75, 1) = 46.875;
	TX1(76, 1) = 51.5625;
	TX1(77, 1) = 56.25;
	TX1(78, 1) = 60.9375;
	TX1(79, 1) = 65.625;
	TX1(80, 1) = 70.3125;
	TX1(81, 1) = 75;
	TX1(82, 1) = 79.6875;
	TX1(83, 1) = 84.375;
	TX1(84, 1) = 89.0625;
	TX1(85, 1) = 93.75;
	TX1(86, 1) = 98.4375;
	TX1(87, 1) = 103.125;
	TX1(88, 1) = 107.8125;
	TX1(89, 1) = 112.5;
	TX1(90, 1) = 117.1875;
	TX1(91, 1) = 121.875;
	TX1(92, 1) = 126.5625;
	TX1(93, 1) = 131.25;
	TX1(94, 1) = 135.9375;
	TX1(95, 1) = 140.625;
	TX1(96, 1) = 145.3125;
	TX1(97, 1) = 150;
	TX1(98, 1) = 154.6875;
	TX1(99, 1) = 159.375;
	TX1(100, 1) = 164.0625;
	TX1(101, 1) = 168.75;
	TX1(102, 1) = 173.4375;
	TX1(103, 1) = 178.125;
	TX1(104, 1) = 182.8125;
	TX1(105, 1) = 187.5;
	TX1(106, 1) = 192.1875;
	TX1(107, 1) = 196.875;
	TX1(108, 1) = 201.5625;
	TX1(109, 1) = 206.25;
	TX1(110, 1) = 210.9375;
	TX1(111, 1) = 215.625;
	TX1(112, 1) = 220.3125;
	TX1(113, 1) = 225;
	TX1(114, 1) = 229.6875;
	TX1(115, 1) = 234.375;
	TX1(116, 1) = 239.0625;
	TX1(117, 1) = 243.75;
	TX1(118, 1) = 248.4375;
	TX1(119, 1) = 253.125;
	TX1(120, 1) = 257.8125;
	TX1(121, 1) = 262.5;
	TX1(122, 1) = 267.1875;
	TX1(123, 1) = 271.875;
	TX1(124, 1) = 276.5625;
	TX1(125, 1) = 281.25;
	TX1(126, 1) = 285.9375;
	TX1(127, 1) = 290.625;
	TX1(128, 1) = 295.3125;
	TX1(129, 1) = 300;
	TX1(130, 1) = 0;
	TX1(131, 1) = 4.6875;
	TX1(132, 1) = 9.375;
	TX1(133, 1) = 14.0625;
	TX1(134, 1) = 18.75;
	TX1(135, 1) = 23.4375;
	TX1(136, 1) = 28.125;
	TX1(137, 1) = 32.8125;
	TX1(138, 1) = 37.5;
	TX1(139, 1) = 42.1875;
	TX1(140, 1) = 46.875;
	TX1(141, 1) = 51.5625;
	TX1(142, 1) = 56.25;
	TX1(143, 1) = 60.9375;
	TX1(144, 1) = 65.625;
	TX1(145, 1) = 70.3125;
	TX1(146, 1) = 75;
	TX1(147, 1) = 79.6875;
	TX1(148, 1) = 84.375;
	TX1(149, 1) = 89.0625;
	TX1(150, 1) = 93.75;
	TX1(151, 1) = 98.4375;
	TX1(152, 1) = 103.125;
	TX1(153, 1) = 107.8125;
	TX1(154, 1) = 112.5;
	TX1(155, 1) = 117.1875;
	TX1(156, 1) = 121.875;
	TX1(157, 1) = 126.5625;
	TX1(158, 1) = 131.25;
	TX1(159, 1) = 135.9375;
	TX1(160, 1) = 140.625;
	TX1(161, 1) = 145.3125;
	TX1(162, 1) = 150;
	TX1(163, 1) = 154.6875;
	TX1(164, 1) = 159.375;
	TX1(165, 1) = 164.0625;
	TX1(166, 1) = 168.75;
	TX1(167, 1) = 173.4375;
	TX1(168, 1) = 178.125;
	TX1(169, 1) = 182.8125;
	TX1(170, 1) = 187.5;
	TX1(171, 1) = 192.1875;
	TX1(172, 1) = 196.875;
	TX1(173, 1) = 201.5625;
	TX1(174, 1) = 206.25;
	TX1(175, 1) = 210.9375;
	TX1(176, 1) = 215.625;
	TX1(177, 1) = 220.3125;
	TX1(178, 1) = 225;
	TX1(179, 1) = 229.6875;
	TX1(180, 1) = 234.375;
	TX1(181, 1) = 239.0625;
	TX1(182, 1) = 243.75;
	TX1(183, 1) = 248.4375;
	TX1(184, 1) = 253.125;
	TX1(185, 1) = 257.8125;
	TX1(186, 1) = 262.5;
	TX1(187, 1) = 267.1875;
	TX1(188, 1) = 271.875;
	TX1(189, 1) = 276.5625;
	TX1(190, 1) = 281.25;
	TX1(191, 1) = 285.9375;
	TX1(192, 1) = 290.625;
	TX1(193, 1) = 295.3125;
	TX1(194, 1) = 300;

	return TX1;
}

MatrixXd GetTX2()
{
	MatrixXd TX1(15, 2);

	TX1(0, 0) = 0;
	TX1(1, 0) = 0;
	TX1(2, 0) = 0;
	TX1(3, 0) = 0;
	TX1(4, 0) = 0;
	TX1(5, 0) = 0.432499999999999;
	TX1(6, 0) = 0.432499999999999;
	TX1(7, 0) = 0.432499999999999;
	TX1(8, 0) = 0.432499999999999;
	TX1(9, 0) = 0.432499999999999;
	TX1(10, 0) = 0.864999999999999;
	TX1(11, 0) = 0.864999999999999;
	TX1(12, 0) = 0.864999999999999;
	TX1(13, 0) = 0.864999999999999;
	TX1(14, 0) = 0.864999999999999;
	TX1(0, 1) = 0;
	TX1(1, 1) = 75;
	TX1(2, 1) = 150;
	TX1(3, 1) = 225;
	TX1(4, 1) = 300;
	TX1(5, 1) = 0;
	TX1(6, 1) = 75;
	TX1(7, 1) = 150;
	TX1(8, 1) = 225;
	TX1(9, 1) = 300;
	TX1(10, 1) = 0;
	TX1(11, 1) = 75;
	TX1(12, 1) = 150;
	TX1(13, 1) = 225;
	TX1(14, 1) = 300;
	return TX1;
}

MatrixXd GetTX2_2Asset()
{
	MatrixXd TX1(81, 3);

	TX1(0, 0) = 0;
	TX1(1, 0) = 0;
	TX1(2, 0) = 0;
	TX1(3, 0) = 0;
	TX1(4, 0) = 0;
	TX1(5, 0) = 0;
	TX1(6, 0) = 0;
	TX1(7, 0) = 0;
	TX1(8, 0) = 0;
	TX1(9, 0) = 0;
	TX1(10, 0) = 0;
	TX1(11, 0) = 0;
	TX1(12, 0) = 0;
	TX1(13, 0) = 0;
	TX1(14, 0) = 0;
	TX1(15, 0) = 0;
	TX1(16, 0) = 0;
	TX1(17, 0) = 0;
	TX1(18, 0) = 0;
	TX1(19, 0) = 0;
	TX1(20, 0) = 0;
	TX1(21, 0) = 0;
	TX1(22, 0) = 0;
	TX1(23, 0) = 0;
	TX1(24, 0) = 0;
	TX1(25, 0) = 0;
	TX1(26, 0) = 0;
	TX1(27, 0) = 0.456;
	TX1(28, 0) = 0.456;
	TX1(29, 0) = 0.456;
	TX1(30, 0) = 0.456;
	TX1(31, 0) = 0.456;
	TX1(32, 0) = 0.456;
	TX1(33, 0) = 0.456;
	TX1(34, 0) = 0.456;
	TX1(35, 0) = 0.456;
	TX1(36, 0) = 0.456;
	TX1(37, 0) = 0.456;
	TX1(38, 0) = 0.456;
	TX1(39, 0) = 0.456;
	TX1(40, 0) = 0.456;
	TX1(41, 0) = 0.456;
	TX1(42, 0) = 0.456;
	TX1(43, 0) = 0.456;
	TX1(44, 0) = 0.456;
	TX1(45, 0) = 0.456;
	TX1(46, 0) = 0.456;
	TX1(47, 0) = 0.456;
	TX1(48, 0) = 0.456;
	TX1(49, 0) = 0.456;
	TX1(50, 0) = 0.456;
	TX1(51, 0) = 0.456;
	TX1(52, 0) = 0.456;
	TX1(53, 0) = 0.456;
	TX1(54, 0) = 0.911999999999999;
	TX1(55, 0) = 0.911999999999999;
	TX1(56, 0) = 0.911999999999999;
	TX1(57, 0) = 0.911999999999999;
	TX1(58, 0) = 0.911999999999999;
	TX1(59, 0) = 0.911999999999999;
	TX1(60, 0) = 0.911999999999999;
	TX1(61, 0) = 0.911999999999999;
	TX1(62, 0) = 0.911999999999999;
	TX1(63, 0) = 0.911999999999999;
	TX1(64, 0) = 0.911999999999999;
	TX1(65, 0) = 0.911999999999999;
	TX1(66, 0) = 0.911999999999999;
	TX1(67, 0) = 0.911999999999999;
	TX1(68, 0) = 0.911999999999999;
	TX1(69, 0) = 0.911999999999999;
	TX1(70, 0) = 0.911999999999999;
	TX1(71, 0) = 0.911999999999999;
	TX1(72, 0) = 0.911999999999999;
	TX1(73, 0) = 0.911999999999999;
	TX1(74, 0) = 0.911999999999999;
	TX1(75, 0) = 0.911999999999999;
	TX1(76, 0) = 0.911999999999999;
	TX1(77, 0) = 0.911999999999999;
	TX1(78, 0) = 0.911999999999999;
	TX1(79, 0) = 0.911999999999999;
	TX1(80, 0) = 0.911999999999999;
	TX1(0, 1) = 0;
	TX1(1, 1) = 0;
	TX1(2, 1) = 0;
	TX1(3, 1) = 0;
	TX1(4, 1) = 0;
	TX1(5, 1) = 0;
	TX1(6, 1) = 0;
	TX1(7, 1) = 0;
	TX1(8, 1) = 0;
	TX1(9, 1) = 150;
	TX1(10, 1) = 150;
	TX1(11, 1) = 150;
	TX1(12, 1) = 150;
	TX1(13, 1) = 150;
	TX1(14, 1) = 150;
	TX1(15, 1) = 150;
	TX1(16, 1) = 150;
	TX1(17, 1) = 150;
	TX1(18, 1) = 300;
	TX1(19, 1) = 300;
	TX1(20, 1) = 300;
	TX1(21, 1) = 300;
	TX1(22, 1) = 300;
	TX1(23, 1) = 300;
	TX1(24, 1) = 300;
	TX1(25, 1) = 300;
	TX1(26, 1) = 300;
	TX1(27, 1) = 0;
	TX1(28, 1) = 0;
	TX1(29, 1) = 0;
	TX1(30, 1) = 0;
	TX1(31, 1) = 0;
	TX1(32, 1) = 0;
	TX1(33, 1) = 0;
	TX1(34, 1) = 0;
	TX1(35, 1) = 0;
	TX1(36, 1) = 150;
	TX1(37, 1) = 150;
	TX1(38, 1) = 150;
	TX1(39, 1) = 150;
	TX1(40, 1) = 150;
	TX1(41, 1) = 150;
	TX1(42, 1) = 150;
	TX1(43, 1) = 150;
	TX1(44, 1) = 150;
	TX1(45, 1) = 300;
	TX1(46, 1) = 300;
	TX1(47, 1) = 300;
	TX1(48, 1) = 300;
	TX1(49, 1) = 300;
	TX1(50, 1) = 300;
	TX1(51, 1) = 300;
	TX1(52, 1) = 300;
	TX1(53, 1) = 300;
	TX1(54, 1) = 0;
	TX1(55, 1) = 0;
	TX1(56, 1) = 0;
	TX1(57, 1) = 0;
	TX1(58, 1) = 0;
	TX1(59, 1) = 0;
	TX1(60, 1) = 0;
	TX1(61, 1) = 0;
	TX1(62, 1) = 0;
	TX1(63, 1) = 150;
	TX1(64, 1) = 150;
	TX1(65, 1) = 150;
	TX1(66, 1) = 150;
	TX1(67, 1) = 150;
	TX1(68, 1) = 150;
	TX1(69, 1) = 150;
	TX1(70, 1) = 150;
	TX1(71, 1) = 150;
	TX1(72, 1) = 300;
	TX1(73, 1) = 300;
	TX1(74, 1) = 300;
	TX1(75, 1) = 300;
	TX1(76, 1) = 300;
	TX1(77, 1) = 300;
	TX1(78, 1) = 300;
	TX1(79, 1) = 300;
	TX1(80, 1) = 300;
	TX1(0, 2) = 0;
	TX1(1, 2) = 37.5;
	TX1(2, 2) = 75;
	TX1(3, 2) = 112.5;
	TX1(4, 2) = 150;
	TX1(5, 2) = 187.5;
	TX1(6, 2) = 225;
	TX1(7, 2) = 262.5;
	TX1(8, 2) = 300;
	TX1(9, 2) = 0;
	TX1(10, 2) = 37.5;
	TX1(11, 2) = 75;
	TX1(12, 2) = 112.5;
	TX1(13, 2) = 150;
	TX1(14, 2) = 187.5;
	TX1(15, 2) = 225;
	TX1(16, 2) = 262.5;
	TX1(17, 2) = 300;
	TX1(18, 2) = 0;
	TX1(19, 2) = 37.5;
	TX1(20, 2) = 75;
	TX1(21, 2) = 112.5;
	TX1(22, 2) = 150;
	TX1(23, 2) = 187.5;
	TX1(24, 2) = 225;
	TX1(25, 2) = 262.5;
	TX1(26, 2) = 300;
	TX1(27, 2) = 0;
	TX1(28, 2) = 37.5;
	TX1(29, 2) = 75;
	TX1(30, 2) = 112.5;
	TX1(31, 2) = 150;
	TX1(32, 2) = 187.5;
	TX1(33, 2) = 225;
	TX1(34, 2) = 262.5;
	TX1(35, 2) = 300;
	TX1(36, 2) = 0;
	TX1(37, 2) = 37.5;
	TX1(38, 2) = 75;
	TX1(39, 2) = 112.5;
	TX1(40, 2) = 150;
	TX1(41, 2) = 187.5;
	TX1(42, 2) = 225;
	TX1(43, 2) = 262.5;
	TX1(44, 2) = 300;
	TX1(45, 2) = 0;
	TX1(46, 2) = 37.5;
	TX1(47, 2) = 75;
	TX1(48, 2) = 112.5;
	TX1(49, 2) = 150;
	TX1(50, 2) = 187.5;
	TX1(51, 2) = 225;
	TX1(52, 2) = 262.5;
	TX1(53, 2) = 300;
	TX1(54, 2) = 0;
	TX1(55, 2) = 37.5;
	TX1(56, 2) = 75;
	TX1(57, 2) = 112.5;
	TX1(58, 2) = 150;
	TX1(59, 2) = 187.5;
	TX1(60, 2) = 225;
	TX1(61, 2) = 262.5;
	TX1(62, 2) = 300;
	TX1(63, 2) = 0;
	TX1(64, 2) = 37.5;
	TX1(65, 2) = 75;
	TX1(66, 2) = 112.5;
	TX1(67, 2) = 150;
	TX1(68, 2) = 187.5;
	TX1(69, 2) = 225;
	TX1(70, 2) = 262.5;
	TX1(71, 2) = 300;
	TX1(72, 2) = 0;
	TX1(73, 2) = 37.5;
	TX1(74, 2) = 75;
	TX1(75, 2) = 112.5;
	TX1(76, 2) = 150;
	TX1(77, 2) = 187.5;
	TX1(78, 2) = 225;
	TX1(79, 2) = 262.5;
	TX1(80, 2) = 300;

	return TX1;
}

MatrixXd GetTX1_2Asset()
{
	MatrixXd TX1(27, 2);

	TX1(0, 0) = 0;
	TX1(1, 0) = 0;
	TX1(2, 0) = 0;
	TX1(3, 0) = 0;
	TX1(4, 0) = 0;
	TX1(5, 0) = 0;
	TX1(6, 0) = 0;
	TX1(7, 0) = 0;
	TX1(8, 0) = 0;
	TX1(9, 0) = 0.4665;
	TX1(10, 0) = 0.4665;
	TX1(11, 0) = 0.4665;
	TX1(12, 0) = 0.4665;
	TX1(13, 0) = 0.4665;
	TX1(14, 0) = 0.4665;
	TX1(15, 0) = 0.4665;
	TX1(16, 0) = 0.4665;
	TX1(17, 0) = 0.4665;
	TX1(18, 0) = 0.932999999999999;
	TX1(19, 0) = 0.932999999999999;
	TX1(20, 0) = 0.932999999999999;
	TX1(21, 0) = 0.932999999999999;
	TX1(22, 0) = 0.932999999999999;
	TX1(23, 0) = 0.932999999999999;
	TX1(24, 0) = 0.932999999999999;
	TX1(25, 0) = 0.932999999999999;
	TX1(26, 0) = 0.932999999999999;
	TX1(0, 1) = 0;
	TX1(1, 1) = 37.5;
	TX1(2, 1) = 75;
	TX1(3, 1) = 112.5;
	TX1(4, 1) = 150;
	TX1(5, 1) = 187.5;
	TX1(6, 1) = 225;
	TX1(7, 1) = 262.5;
	TX1(8, 1) = 300;
	TX1(9, 1) = 0;
	TX1(10, 1) = 37.5;
	TX1(11, 1) = 75;
	TX1(12, 1) = 112.5;
	TX1(13, 1) = 150;
	TX1(14, 1) = 187.5;
	TX1(15, 1) = 225;
	TX1(16, 1) = 262.5;
	TX1(17, 1) = 300;
	TX1(18, 1) = 0;
	TX1(19, 1) = 37.5;
	TX1(20, 1) = 75;
	TX1(21, 1) = 112.5;
	TX1(22, 1) = 150;
	TX1(23, 1) = 187.5;
	TX1(24, 1) = 225;
	TX1(25, 1) = 262.5;
	TX1(26, 1) = 300;


	return TX1;
}

MatrixXd GetCN1_2Asset()
{
	MatrixXd CN1(15, 2);

	CN1(0, 0) = 0;
	CN1(1, 0) = 0;
	CN1(2, 0) = 0;
	CN1(3, 0) = 0;
	CN1(4, 0) = 0;
	CN1(5, 0) = 0.4665;
	CN1(6, 0) = 0.4665;
	CN1(7, 0) = 0.4665;
	CN1(8, 0) = 0.4665;
	CN1(9, 0) = 0.4665;
	CN1(10, 0) = 0.932999999999999;
	CN1(11, 0) = 0.932999999999999;
	CN1(12, 0) = 0.932999999999999;
	CN1(13, 0) = 0.932999999999999;
	CN1(14, 0) = 0.932999999999999;
	CN1(0, 1) = 0;
	CN1(1, 1) = 75;
	CN1(2, 1) = 150;
	CN1(3, 1) = 225;
	CN1(4, 1) = 300;
	CN1(5, 1) = 0;
	CN1(6, 1) = 75;
	CN1(7, 1) = 150;
	CN1(8, 1) = 225;
	CN1(9, 1) = 300;
	CN1(10, 1) = 0;
	CN1(11, 1) = 75;
	CN1(12, 1) = 150;
	CN1(13, 1) = 225;
	CN1(14, 1) = 300;



	return CN1;
}

void TestRbfInterpolationPDE_2Asset()
{
	
	MatrixXd TX1 = GetTX1_2Asset();
	MatrixXd CN = GetCN1_2Asset();

	MatrixXd C(1, 2);
	MatrixXd A(1, 2);
	C << 1.866, 600;
	A << 2, 4; //TX2
	
	Leicester::ThrustLib::GaussianNd1 cGaussian(TX1);
	for (int i = 0; i < 1; i++)
	{
		printf("i=%i\r\n", i);
		vector<MatrixXd> res = cGaussian.GaussianNd(CN,A, C);
		//vector<MatrixXd> res = ThrustLib::Gaussian::Gaussian2d(TX1, CN, A, C);
		//wcout << "Phi1:" << endl;
		//wcout << printMatrix(res[0].col(0)) << endl;
		//wcout << "Phi2:" << endl;
		//wcout << printMatrix(res[1].col(0)) << endl;
		wcout << "D:" << endl;
		wcout << Utility::printMatrix(res[0].col(1)) << endl;
		wcout << "Dt:" << endl;
		wcout << Utility::printMatrix(res[1].col(1)) << endl;
		wcout << "Dx:" << endl;
		wcout << Utility::printMatrix(res[2].col(1)) << endl;
		wcout << "Dxx:" << endl;
		wcout << Utility::printMatrix(res[3].col(1)) << endl;
	}
}

void TestRbfInterpolation_2Asset()
{
	//MatrixXd TX1 = GetTX7();
	//MatrixXd CN = GetTX7();
	MatrixXd TX1 = GetTX2_2Asset();
	MatrixXd CN = GetTX2_2Asset();

	MatrixXd C(1, 3);
	MatrixXd A(1, 3);
	C << 1.824,	600, 600;
	//A << 2, 64; //TX7
	A << 2,	2, 8; //TX2
	MatrixXd D(TX1.rows(), TX1.rows());
	Leicester::ThrustLib::GaussianNd1 cGaussian(TX1, CN);
	for (int i = 0; i < 1; i++)
	{
		printf("i=%i\r\n", i);
		vector<MatrixXd> res = cGaussian.GaussianNd(A, C);
		//vector<MatrixXd> res = ThrustLib::Gaussian::Gaussian2d(TX1, CN, A, C);
		//wcout << "Phi1:" << endl;
		//wcout << printMatrix(res[0].col(0)) << endl;
		//wcout << "Phi2:" << endl;
		//wcout << printMatrix(res[1].col(0)) << endl;
		//wcout << "D:" << endl;
		//wcout << Utility::printMatrix(res[0].col(0)) << endl;
		//wcout << "Dt:" << endl;
		//wcout << Utility::printMatrix(res[1].col(0)) << endl;
		//wcout << "Dx:" << endl;
		//wcout << Utility::printMatrix(res[2].col(0)) << endl;
		wcout << "Dxx:" << endl;
		wcout << Utility::printMatrix(res[3].col(80)) << endl;
	}
}

void TestRbfInterpolation()
{
	//MatrixXd TX1 = GetTX7();
	//MatrixXd CN = GetTX7();
	MatrixXd TX1 = GetTX2();
	MatrixXd CN = GetTX2();

	MatrixXd C(1, 2);
	MatrixXd A(1, 2);
	C << 1.73, 600;
	//A << 2, 64; //TX7
	A << 2, 4; //TX2
	MatrixXd D(TX1.rows(), TX1.rows());
	Leicester::ThrustLib::Gaussian cGaussian(TX1, TX1);
	for (int i = 0; i < 10; i++)
	{
		printf("i=%i\r\n", i);
		vector<MatrixXd> res = cGaussian.Gaussian2d(A, C);
		//vector<MatrixXd> res = ThrustLib::Gaussian::Gaussian2d(TX1, CN, A, C);
		//wcout << "Phi1:" << endl;
		//wcout << printMatrix(res[0].col(0)) << endl;
		//wcout << "Phi2:" << endl;
		//wcout << printMatrix(res[1].col(0)) << endl;
		wcout << "D:" << endl;
		wcout << Utility::printMatrix(res[0].col(0)) << endl;
		wcout << "Dt:" << endl;
		wcout << Utility::printMatrix(res[1].col(0)) << endl;
		wcout << "Dx:" << endl;
		wcout << Utility::printMatrix(res[2].col(0)) << endl;
		wcout << "Dxx:" << endl;
		wcout << Utility::printMatrix(res[3].col(0)) << endl;
	}
}

void TestRbfInterpolation_2()
{
	//MatrixXd TX1 = GetTX7();
	//MatrixXd CN = GetTX7();
	MatrixXd TX1 = GetTX2();
	MatrixXd CN = GetTX2();

	MatrixXd C(1, 2);
	MatrixXd A(1, 2);
	C << 1.73, 600;
	//A << 2, 64; //TX7
	A << 2, 4; //TX2
	MatrixXd D(TX1.rows(), TX1.rows());
	Leicester::ThrustLib::Gaussian cGaussian;
	for (int i = 0; i < 10; i++)
	{
		printf("i=%i\r\n", i);
		double N[4] = { 1,2,3,5 };

		vector<MatrixXd> res = cGaussian.Gaussian2d_2(0, 0.8650, N, A, C);

		//vector<MatrixXd> res = ThrustLib::Gaussian::Gaussian2d(TX1, CN, A, C);
		//wcout << "Phi1:" << endl;
		//wcout << printMatrix(res[0].col(0)) << endl;
		//wcout << "Phi2:" << endl;
		//wcout << printMatrix(res[1].col(0)) << endl;
		wcout << "D:" << endl;
		wcout << Utility::printMatrix(res[0].col(0)) << endl;
		wcout << "Dt:" << endl;
		wcout << Utility::printMatrix(res[1].col(0)) << endl;
		wcout << "Dx:" << endl;
		wcout << Utility::printMatrix(res[2].col(0)) << endl;
		wcout << "Dxx:" << endl;
		wcout << Utility::printMatrix(res[3].col(0)) << endl;
	}
}

//void printMatrix_CUDA(double *matrix, dim3 dimMatrix)
//{
//
//	printf("printing matrix data=");
//	for (int x = 0; x < 2 + dimMatrix.x * dimMatrix.y; x++)
//		printf("%f,", matrix[x]);
//	printf("\r\n");
//	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);
//
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
//
//}
//
//
//void subnumber(int b, int d, double matrix[])
//{
//
//	double *L = NULL;
//	if (d == 1)
//	{
//		double * l = (double*)malloc(3 * sizeof(double));
//		l[0] = 1;
//		l[1] = 1;
//		l[2] = b;
//		L = l;
//	}
//	else
//	{
//		int nbot = 1;
//
//		int Lrows = 0;
//		int Lcols = 0;
//		for (int i = 0; i < b - d + 1; i++)
//		{
//			double* indextemp = (double*)malloc(512 * sizeof(double));
//
//			subnumber(b - (i + 1), d - 1, indextemp);
//			//printMatrix_CUDA(indextemp, dim3(indextemp[0], indextemp[1]));
//
//			int s = indextemp[0];
//			int ntop = nbot + s - 1;
//
//			double*l = (double*)malloc((ntop*d + 2)* sizeof(double));
//
//			l[0] = ntop;
//			l[1] = d;
//			double *ones = (double*)malloc((s+2) * sizeof(double));
//			ones[0] = s;
//			ones[1] = 1;
//
//			thrust::fill(thrust::seq, ones + 2, ones + 2 + s, (i + 1));
//
//			int start = nbot;
//			int end = start + ntop - nbot;
//
//			//fill the first column with 'ones'
//			//thrust::fill(thrust::seq, l + 2 + start, l + 2 + end, (i + 1));
//			//fill the rest with 'indextemp'
//			//thrust::copy(thrust::seq, indextemp + 2, indextemp + 2 + (int)(l[0] * l[1]) - 1, l + start + ntop);
//			int jMin = 0;
//			int increment = 1;
//			if (L != NULL)
//			{
//				int count = 0;
//				for (int x = 0; x < L[1]; x++)
//					for (int y = 0; y < L[0]; y++)
//					{
//						int diff = l[0] - L[0];
//						l[count + 2 + x * diff] = L[count + 2];
//						//int indx = (x * L[0]) + y + 2;
//
//						//l[indx] = L[indx];
//						count++;
//					}
//				jMin = L[0];
//				increment = L[0];
//			}
//
//			int rows = l[0];
//			int cols = l[1];
//			int k = 0;
//			for (int j = jMin; j < rows * cols; j+=rows, k++)
//			{
//				int indx = j + 2;
//				if (j -jMin < rows)//first col
//					l[indx] = i + 1;
//				else
//					l[indx] = indextemp[k+1];
//			}
//
//			nbot = ntop + 1;
//
//			//if (Lrows > 0)
//			//{
//			//	thrust::copy(thrust::seq, L, L + (Lrows * Lcols) - 1, l);
//			//}
//			L = (double*)malloc(sizeof(double) * (l[0] * l[1] + 2));
//			for (int i = 0; i < (int)(l[0] * l[1]) + 2; i++)
//				L[i] = l[i];
//			Lrows = ntop;
//			Lcols = d;
//		}
//	}
//	for (int i = 0; i < (int)(L[0] * L[1] )+2; i++)
//		matrix[i] = L[i];
//
//}
//
//void Add_CUDA(int b, int d, double N[])
//{
//	double *d_L = (double*)malloc(3 * sizeof(double));
//	d_L[0] = 1;
//	d_L[1] = 1;
//
//	subnumber(b, d, d_L);
//	int ch = d_L[0];
//	//free(N);
//
//	//N = (double*)malloc((2 + ch * d) * sizeof(double));
//	N[0] = ch;
//	N[1] = d;
//	int idx = 2;
//	for (int i = 0; i < ch; i++)
//		for (int j = 0; j < d; j++, idx++)
//		{
//			//int idx = 2 + j + (i * ch);
//			N[idx] = pow(2, d_L[idx])  + 1;
//		}
//
//}
//
//double* GetColumn(double matrix[], int col)
//{
//	double* result = (double*)malloc((2 + matrix[0]) * sizeof(double));
//	int columnStart = 2 + ((matrix[0] + 2) * col);
//	result[0] = matrix[columnStart];
//	result[1] = 1;
//	for (int i = 0; i < result[0]; i++)
//	{
//		int idx = i + 2 + columnStart;
//		result[i + 2] = matrix[idx];
//	}
//	return result;
//}
//
//void SetColumn(double matrix[], double vector[], int col)
//{
//	/*double* result = (double*)malloc((2 + matrix[0] * matrix[1]) * sizeof(double));
//	result[0] = matrix[0];
//	result[1] = matrix[1];*/
//	for (int i = 0; i < vector[0]; i++)
//	{
//		int idx = i + (matrix[0] * col);
//		matrix[idx + 2] = vector[i + 2];
//	}
//	//return result;
//}
//
//double* GetRow(double matrix[], int row)
//{
//	double* result = (double*)malloc((2 + matrix[1]) * sizeof(double));
//	result[0] = 1;
//	result[1] = matrix[1];
//	int rowIdx = 0;
//	for (int i = 0; i < matrix[0] * matrix[1]; i++)
//	{
//		if ((i % (int)matrix[0]) == row)
//		{
//			result[rowIdx+2] = matrix[i + 2];
//			rowIdx++;
//		}
//	}
//	return result;
//}
//
//double* ReplicateN(double linearVector[], double totalLength, int dups)
//{
//	double* Result = (double*)malloc((2+totalLength) * sizeof(double));
//	Result[0] = totalLength;
//	Result[1] = 1;
//	int size = linearVector[0] * linearVector[1];
//	for (int i = 0; i < totalLength; i += (size * dups))
//	{
//		for (int j = 0; j < size; j++)
//		{
//			for (int duplicated = 0; duplicated < dups; duplicated++)
//			{
//				int idx = i + (j * dups) + duplicated;
//				if (idx < totalLength)
//				{
//					Result[idx+2] = linearVector[j+2];
//					//cout << "idx="<< idx << " v[j]=" << v[j] << endl;
//				}
//
//			}
//		}
//	}
//	return Result;
//}
//
//double Max(double matrix[])
//{
//	double max = -1;
//	double length = 2 + matrix[0] * matrix[1];
//	for (int i = 2; i < length; i++)
//		if (matrix[i] > max)
//			max = matrix[i];
//	return max;
//}
//
//double* VectorLinSpaced(int i, double lowerLimit, double upperLimit)
//{
//	double difference = upperLimit - lowerLimit;
//	double dx = difference / (i-1);
//	double* result = (double*)malloc((i + 2)* sizeof(double));
//	result[0] = i;
//	result[1] = 1;
//	for(int j=0; j < i; j++)
//	{
//		result[j+2] = lowerLimit + (j * dx);
//	}
//	return result;
//}
//
//double* GenerateTestNodes(double timeLowerLimit, double timeUpperLimit, double lowerLimits[], double upperLimits[], double N[])
//{
//	//vector<VectorXd> linearGrid;
//
//	int product = 1;
//	int nCols = N[1];
//	int nRows = N[0];
//	double max = Max(N);
//	double* linearGrid = (double*)malloc((2 + (nCols * (2 + max))) * sizeof(double)); // this is an array of NCols vectors of max-rows each
//	linearGrid[0] = max;
//	linearGrid[1] = nCols;
//	for (int n = 0; n < nCols; n++) //N.Cols() is #dimensions
//	{
//		int idx = n * nRows;
//		int i = N[idx+2];
//		product *= i;
//
//		//VectorXd linearDimension;
//		double* linearDimension;
//		if (n == 0)
//			linearDimension = VectorLinSpaced(i, timeLowerLimit, timeUpperLimit);
//		else
//			linearDimension = VectorLinSpaced(i, lowerLimits[n - 1], upperLimits[n - 1]);
//		double length = linearDimension[0] * linearDimension[1] + 2;
//		//linearGrid.push_back(linearDimension);
//		for (int j = 0; j < length; j++)
//		{
//			int idx = 2 + n* (linearGrid[0]+2);
//			linearGrid[idx+j] = linearDimension[j];
//		}
//
//	}
//
//
//	//MatrixXd TXYZ(product, N.cols());
//	double* TXYZ = (double*)malloc(product * nCols * sizeof(double));
//	TXYZ[0] = product;
//	TXYZ[1] = nCols;
//	int dimension = 0;
//	int dups = 1;
//	for (int col = 0; col < nCols; col++)
//	{
//		int idx = 2 + dimension * (2 +linearGrid[0]);
//		if (dimension == 0)
//		{
//			dups = product / linearGrid[idx];
//		}
//		else
//			dups = dups / linearGrid[idx];
//		double* linearVector = GetColumn(linearGrid, col);
//		double* column = ReplicateN(linearVector, product, dups);
//		SetColumn(TXYZ, column, col);
//		dimension++;
//	}
//	return TXYZ;
//}
void f(int array[]) {

	array[0] = 4;
	array[1] = 5;
	array[2] = 6;
}

void TestArrayCopy()
{
	//cublasHandle_t handle;
	cudaError_t cudaerr;
	cudaEvent_t start, stop;
	//cublasStatus_t stat;
	const double alpha = 1.0f;
	const double beta = 0.0f;

	double *h_A = new double[5];
	double *h_B = new double[5];
	double *h_C = new double[6];
	for (int i = 0; i < 5; i++)
	{
		h_A[i] = i;
		h_B[i] = i;
	}



	double **h_AA, **h_BB, **h_CC;
	h_AA = (double**)malloc(6 * sizeof(double*));
	h_BB = (double**)malloc(6 * sizeof(double*));
	h_CC = (double**)malloc(6 * sizeof(double*));
	for (int i = 0; i < 6; i++) {
		cudaMalloc((void **)&h_AA[i], 5 * sizeof(double));
		cudaMalloc((void **)&h_BB[i], 5 * sizeof(double));
		cudaMalloc((void **)&h_CC[i], sizeof(double));
		cudaMemcpy(h_AA[i], h_A, 5 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(h_BB[i], h_B, 5 * sizeof(double), cudaMemcpyHostToDevice);
	}
	double **d_AA, **d_BB, **d_CC;
	cudaMalloc(&d_AA, 6 * sizeof(double*));
	cudaMalloc(&d_BB, 6 * sizeof(double*));
	cudaMalloc(&d_CC, 6 * sizeof(double*));
	cudaerr = cudaMemcpy(d_AA, h_AA, 6 * sizeof(double*), cudaMemcpyHostToDevice);
	cudaerr = cudaMemcpy(d_BB, h_BB, 6 * sizeof(double*), cudaMemcpyHostToDevice);
	cudaerr = cudaMemcpy(d_CC, h_CC, 6 * sizeof(double*), cudaMemcpyHostToDevice);
	
	Leicester::ThrustLib::BuildRegistry << <1, 1 >> >(3, 2, 0, 0.5, 0, 300, d_CC);

	cudaerr = cudaMemcpy(h_CC, d_CC, sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 6; i++)
		cudaMemcpy(h_C + i, h_CC[i], sizeof(double), cudaMemcpyDeviceToHost);
	//cublasDestroy(handle);

	double nNodes = *h_C;

	//for (int i = 1; i < nNodes; i++)
	//{
	//	int rows = N[2 + i] * N[2 + i + (int)nRows];
	//	thrust::device_ptr<double> d_ptr(h_nodes[i]);
	//	const thrust::device_vector<double> d_v(d_ptr, d_ptr + (rows * 2));//TODO: change 2 to dimensions
	//	nodesDetails nd;
	//	nd.rows = rows;
	//	nd.cols = 2;
	//	nd.nodes = d_v;
	//	this->nodeMap[i] = nd;
	//}
}

int main()
{
	cudaDeviceProp p;
	cudaError_t e= cudaGetDeviceProperties(&p, 0);
	int max = p.maxThreadsPerMultiProcessor;

	//NodeRegistry nr;
	//nr.Add(3,2);

	//double * expected = new double[4]{1,2,2,1};
	//
	//device_vector<double> dactual = nr.Ns["3,2"];

	//double* p_actual = dactual.data().get();
	//double* h_actual = new double[4];

	//cudaError_t e = cudaMemcpy(h_actual, p_actual, sizeof(double) * 4, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	//double* N = (double*)malloc(512 * sizeof(double));
	////double N[512];
	//Add_CUDA(3, 2, N);
	//printMatrix_CUDA(N, dim3(2, 2));
	//double* n = GetRow(N, 0);
	//double lower[1] = {0};
	//double upper[1] = {300};
	//double* TXYZ = GenerateTestNodes(0, 0.8650, lower, upper, n );
	//printMatrix_CUDA(TXYZ, dim3(TXYZ[1], TXYZ[0]));

	//double* N1 = (double*)malloc(512 * sizeof(double));
	//Add_CUDA(4, 2, N1); // 1,2,3,3,2,1
	//n = GetRow(N1, 0);
	//TXYZ = GenerateTestNodes(0, 0.8650, lower, upper, n);
	//printMatrix_CUDA(TXYZ, dim3(TXYZ[1], TXYZ[0]));

	//double* N2 = (double*)malloc(512 * sizeof(double));
	//Add_CUDA(5, 2, N2); // 1,2,3,4,4,3,2,1
	//n = GetRow(N2, 0);
	//TXYZ = GenerateTestNodes(0, 0.8650, lower, upper, n);
	//printMatrix_CUDA(TXYZ, dim3(TXYZ[1], TXYZ[0]));

	//int array[] = { 1,2,3 };

	//f(array);

	//printf("%d %d %d", array[0], array[1], array[2]);
	//TestArrayCopy();
	TestRbfInterpolationPDE_2Asset();
	return 0;
}