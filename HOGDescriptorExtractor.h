#pragma once
#include "use_opencv.h"
class HOGDescriptorExtractor
{
public:
	HOGDescriptorExtractor();
	~HOGDescriptorExtractor();

	void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);
};

