#pragma once

#include <iostream>
#include "use_opencv.h"

namespace VISIONNOOB
{
	namespace PANORAMA
	{
		namespace UTIL
		{
			std::vector<cv::Point> FindLocalMaximal(cv::Mat &src);
			void pointWiseAffineTransform(const cv::Mat &src, cv::Mat& dst, const cv::Mat T);
			void stitch(const cv::Mat& leftImage, const cv::Mat& rightImage, cv::OutputArray dst, const cv::Mat& rightT, int extraVerticalMargin = 30, int extraHorizontalMargin = 30);
			cv::Mat findHomographyWithRANSAC(std::vector<cv::Point2f>& scene, std::vector<cv::Point2f>& obj);
		}
	}
}
