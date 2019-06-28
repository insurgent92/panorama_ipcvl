#include "Matcher.h"

namespace visionNoob
{
	namespace computerVision
	{
		Matcher::Matcher()
		{
		}


		Matcher::~Matcher()
		{
		}

		void Matcher::match(cv::InputArray _descriptors1, cv::InputArray _descriptors2, std::vector<cv::DMatch>& matches)
		{
			cv::Mat descriptors1 = _descriptors1.getMat();
			cv::Mat descriptors2 = _descriptors2.getMat();

			int dimFeature = descriptors1.cols;
			int numFeatures = descriptors2.rows;


			for (int i = 0; i < descriptors1.rows; i++)
			{
				cv::DMatch currentMatch;
				int minDistance = std::numeric_limits<int>::max();
				int queryIdx = 0;
				int trainIdx = 0;

				for (int j = 0; j < descriptors2.rows; j++)
				{
					cv::Mat featureVec1 = descriptors1(cv::Rect(0, i, dimFeature, 1)).clone();
					cv::Mat featureVec2 = descriptors2(cv::Rect(0, j, dimFeature, 1)).clone();

					featureVec1.convertTo(featureVec1, CV_32S);
					featureVec2.convertTo(featureVec2, CV_32S);

					cv::Mat ecuclidianDistance = featureVec1 - featureVec2;
					ecuclidianDistance = ecuclidianDistance.mul(ecuclidianDistance);

					int currentDistance = 0;
					for (int idx = 0; idx < dimFeature; idx++)
					{
						currentDistance += ecuclidianDistance.at<int>(0, idx);
					}

					currentDistance = cv::sqrt(currentDistance); 

					if (currentDistance < minDistance)
					{
						minDistance = currentDistance;
						queryIdx = i;
						trainIdx = j;
					}

				}

				currentMatch.queryIdx = queryIdx;
				currentMatch.trainIdx = trainIdx;
				currentMatch.distance = minDistance;
				matches.push_back(currentMatch);
			}
		}
	}
}

