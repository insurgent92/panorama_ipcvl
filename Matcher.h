#pragma once
#include <iostream>
#include "use_opencv.h"

namespace visionNoob
{
	namespace computerVision
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


