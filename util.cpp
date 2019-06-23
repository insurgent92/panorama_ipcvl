#include <iostream>
#include "use_opencv.h"

namespace VISIONNOOB
{
	namespace PANORAMA
	{
		namespace UTIL
		{
			std::vector<cv::Point> FindLocalMaximal(cv::Mat &src)
			{
				cv::Mat dilated;
				cv::Mat localMax;
				//Size size(5,5);
				//Mat rectKernel = getStructuringElement(MORPH_RECT, size);

				cv::dilate(src, dilated, cv::Mat()); //local max
													 //compare(src, dilated, localMax, cv::CMP_EQ);
				localMax = (src == dilated);
				//imshow("localMax", localMax);

				cv::Mat eroded;
				cv::Mat localMin;
				cv::erode(src, eroded, cv::Mat()); //local min
												   //compare(src, eroded, localMin, CMP_GT);
				localMin = (src > eroded);
				//imshow("localMin", localMin);

				localMax = (localMax & localMin);

				std::vector<cv::Point> points;
				for (int y = 0; y < localMax.rows; y++)
				{
					for (int x = 0; x < localMax.cols; x++)
					{
						uchar uValue = localMax.at<uchar>(y, x);
						if (uValue)
						{
							points.push_back(cv::Point(x, y));
						}
					}
				}
				return points;
			}

			void pointWiseAffineTransform(const cv::Mat &src, cv::Mat& dst, const cv::Mat T)
			{
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i++)
					{
						int x = T.at<double>(0, 0) * i + T.at<double>(0, 1) * j + T.at<double>(0, 2);
						int y = T.at<double>(1, 0) * i + T.at<double>(1, 1) * j + T.at<double>(1, 2);

						dst.at<cv::Vec3b>(y, x)[0] = src.at<cv::Vec3b>(j, i)[0];
						dst.at<cv::Vec3b>(y, x)[1] = src.at<cv::Vec3b>(j, i)[1];
						dst.at<cv::Vec3b>(y, x)[2] = src.at<cv::Vec3b>(j, i)[2];
					}
				}
			}
			void stitch(const cv::Mat& leftImage, const cv::Mat& rightImage, cv::OutputArray dst, const cv::Mat& rightT, int extraVerticalMargin, int extraHorizontalMargin)
			{
				dst.create(cv::Size(leftImage.cols * 2 + extraHorizontalMargin * 2, leftImage.rows + extraVerticalMargin * 2), CV_8UC3);
				cv::Mat dstImage = dst.getMat();
				dstImage.setTo(cv::Scalar(0, 0, 0));
				cv::Mat _rightT = rightT.clone();


				double leftTElements[3][3] = { { 1., 0., (double)extraHorizontalMargin },{ 0., 1., (double)extraVerticalMargin } ,{ 0., 0., 1. } };
				cv::Mat _leftT = cv::Mat(3, 3, CV_64F, &leftTElements);

				_rightT.at<double>(0, 2) += (double)extraHorizontalMargin;
				_rightT.at<double>(1, 2) += (double)extraVerticalMargin;

				pointWiseAffineTransform(leftImage, dstImage, _leftT);
				pointWiseAffineTransform(rightImage, dstImage, _rightT);
			}
		}
	}
}
