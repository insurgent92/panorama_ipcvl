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
		}
	}
}
