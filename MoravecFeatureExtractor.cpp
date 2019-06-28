#include "MoravecFeatureExtractor.h"

namespace visionNoob
{
	namespace computerVision
	{
		MoravecFeatureExtractor::MoravecFeatureExtractor()
		{
		}


		MoravecFeatureExtractor::~MoravecFeatureExtractor()
		{
		}

		void MoravecFeatureExtractor::detect(cv::InputArray _src, std::vector<cv::KeyPoint>& keypoints, int _windowSize, double threshold)
		{
			cornerMORAVEC(_src, keypoints, _windowSize, threshold);
		}

		void MoravecFeatureExtractor::constructKeypoints(cv::Mat keypointsMap, std::vector<cv::KeyPoint>& keypoints)
		{
			for (int idy = 0; idy < keypointsMap.rows; idy++)
			{
				for (int idx = 0; idx < keypointsMap.cols; idx++)
				{
					if (keypointsMap.at<double>(idy, idx) != 0.)
					{
						cv::KeyPoint kpt;
						kpt.pt = cv::Point(idx, idy);
						keypoints.push_back(kpt);
					}
				}
			}
		}

		double MoravecFeatureExtractor::SSD(const cv::Mat& src1, const cv::Mat& src2)
		{
			cv::Mat src_cvt1, src_cvt2;

			src1.convertTo(src_cvt1, CV_64FC1);
			src2.convertTo(src_cvt2, CV_64FC1);

			cv::Mat difference = (src_cvt1 - src_cvt2);
			return (double)cv::sum(difference.mul(difference))[0];
		}

		void MoravecFeatureExtractor::get_prob_features(cv::InputArray _src, int idy, int idx, std::vector<double>& prob_features, int _windowSize)
		{
			cv::Mat src = _src.getMat();
			int roi_size = _windowSize / 2;

			for (int j = idy - roi_size; j <= idy + roi_size; j++)
			{
				for (int i = idx - roi_size; i <= idx + roi_size; i++)
				{
					if (j == idy || i == idx)
						continue;

					if (j < 0 || i < 0 || j >= src.rows || i >= src.cols)
						continue;

					cv::Rect roiRect1 = cv::Rect(idx - roi_size, idy - roi_size, _windowSize, _windowSize);
					cv::Rect roiRect2 = cv::Rect(i - roi_size, j - roi_size, _windowSize, _windowSize);

					if (roiRect1.x < 0 || roiRect1.y < 0 || roiRect1.x + roiRect1.width >= src.cols || roiRect1.y + roiRect1.height >= src.rows)
						continue;

					if (roiRect2.x < 0 || roiRect2.y < 0 || roiRect2.x + roiRect2.width >= src.cols || roiRect2.y + roiRect2.height >= src.rows)
						continue;

					cv::Mat roi1 = src(roiRect1);
					cv::Mat roi2 = src(roiRect2);

					double ssd = SSD(roi1, roi2);
					prob_features.push_back(ssd);
				}
			}
		}


		void MoravecFeatureExtractor::calcKeypointsMap(cv::InputArray _src, cv::OutputArray _dst, int _windowSize, double threshold)
		{
			cv::Mat src = _src.getMat();
			cv::Mat dst(src.size(), CV_64FC1, cv::Scalar(0.0));

			for (int idy = 0; idy < src.rows; idy++)
			{
				for (int idx = 0; idx < src.cols; idx++)
				{

					std::vector<double> prob_features;
					get_prob_features(_src, idy, idx, prob_features, _windowSize);

					if (prob_features.empty())
						continue;

					auto min = std::min_element(std::begin(prob_features), std::end(prob_features));

					if (*min > threshold)
					{

						dst.at<double>(idy, idx) = *min;
					}
				}
			}

			dst.copyTo(_dst);
		}

		void MoravecFeatureExtractor::nonMaximumSuppression(cv::InputArray _src, cv::OutputArray _dst)
		{
			cv::Mat src = _src.getMat();
			cv::Mat dst = src.clone();

			for (int idy = 0; idy < src.rows; idy++)
			{
				for (int idx = 0; idx < src.cols; idx++)
				{
					bool ismaximum = true;

					for (int local_y = -1; local_y <= 1; local_y++)
					{
						for (int local_x = -1; local_x <= 1; local_x++)
						{
							int current_idx = idx + local_x;
							int current_idy = idy + local_y;

							if (current_idx < 0 || current_idx >= src.cols || current_idy < 0 || current_idy >= src.rows)
								continue;

							if (src.at<double>(idy, idx) < src.at<double>(current_idy, current_idx))
								ismaximum = false;
						}
					}

					if (!ismaximum)
						dst.at<double>(idy, idx) = 0.;
				}
			}

			dst.copyTo(_dst);
		}

		void MoravecFeatureExtractor::cornerMORAVEC(cv::InputArray _src, std::vector<cv::KeyPoint>& keypoints, int _windowSize, double threshold)
		{
			cv::Mat keypointsMap;
			calcKeypointsMap(_src, keypointsMap, _windowSize, threshold);
			nonMaximumSuppression(keypointsMap, keypointsMap);
			constructKeypoints(keypointsMap, keypoints);
		}
	}
}
