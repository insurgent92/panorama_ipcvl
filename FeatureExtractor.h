#pragma once
#include <iostream>
#include "use_opencv.h"

namespace VISIONNOOB
{
	namespace PANORAMA
	{
		class FeatureExtractor
		{
		public:
			FeatureExtractor();
			~FeatureExtractor();
			void detect(cv::InputArray _src, std::vector<cv::KeyPoint>& keypoints, int _windowSize, double threshold);
		private:
			void cornerMORAVEC(cv::InputArray _src, std::vector<cv::KeyPoint>& keypoints, int _windowSize, double threshold);
			void calcKeypointsMap(cv::InputArray _src, cv::OutputArray _dst, int _windowSize, double threshold);
			void constructKeypoints(cv::Mat keypointsMap, std::vector<cv::KeyPoint>& keypoints);
			void nonMaximumSuppression(cv::InputArray _src, cv::OutputArray _dst);
			double SSD(const cv::Mat& src1, const cv::Mat& src2);
			void get_prob_features(cv::InputArray _src, int idy, int idx, std::vector<double>& prob_features, int _windowSize);
		};
	}
}


