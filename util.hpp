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
		}
	}
}
