#pragma once
#include "use_opencv.h"
#include "util.hpp"

namespace VISIONNOOB
{
	namespace PANORAMA
	{
		class HOGDescriptorExtractor
		{
		public:
			HOGDescriptorExtractor(
				cv::Size _winSize,
				cv::Size _blockSize,
				cv::Size _blockStride,
				cv::Size _cellSize,
				int _nBins,
				float _L2HysThresh);

			~HOGDescriptorExtractor();

			void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);

		private:
			cv::Mat gradientDirection;
			cv::Mat gradientMagnitude;
			cv::Mat derivativeX;
			cv::Mat detivativeY;

			cv::Size winSize;
			cv::Size blockSize;
			cv::Size blockStride;
			cv::Size cellSize;
			int nBins;
			float L2HysThresh;
		};
	}
}



