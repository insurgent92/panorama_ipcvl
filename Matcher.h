#pragma once
#include <iostream>
#include "use_opencv.h"

namespace VISIONNOOB
{
	namespace PANORAMA
	{
		class Matcher
		{
		public:
			Matcher();
			~Matcher();
			void match(cv::InputArray descriptors1, cv::InputArray descriptors2, std::vector<cv::DMatch>& matches);
		};
	}
}


