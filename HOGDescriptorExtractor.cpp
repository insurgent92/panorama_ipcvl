#include "HOGDescriptorExtractor.h"

namespace VISIONNOOB
{
	namespace PANORAMA
	{
		HOGDescriptorExtractor::HOGDescriptorExtractor()
		{
		}


		HOGDescriptorExtractor::~HOGDescriptorExtractor()
		{
		}

		void HOGDescriptorExtractor::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray _descriptors)
		{
			int dimVector = 32; 
			cv::Mat descriptors(cv::Size(dimVector, keypoints.size()), CV_8UC1, cv::Scalar(0));
			for (int i = 0 ; i < keypoints.size() ; i ++)
			{
				cv::Mat featureVector;

				/* Todo make feature*/
				//see http://blog.naver.com/PostView.nhn?blogId=tommybee&logNo=221173056260&parentCategoryNo=&categoryNo=57&viewDate=&isShowPopularPosts=true&from=search

				cv::Mat roi = descriptors(cv::Rect(0, i, dimVector, 1));
				featureVector.copyTo(roi);
			}
		}
	}
}
