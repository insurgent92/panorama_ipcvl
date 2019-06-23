#include <iostream>
#include <float.h>
#include <cstdlib>
#include <ctime>
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
						int x = T.at<double>(0, 0) * (double)i + T.at<double>(0, 1) * (double)j + T.at<double>(0, 2);
						int y = T.at<double>(1, 0) * (double)i + T.at<double>(1, 1) * (double)j + T.at<double>(1, 2);

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

			cv::Mat estimateLeastSquare(std::vector<std::tuple<cv::Point2f, cv::Point2f>> keypointsPair)
			{
				cv::Mat M(cv::Size(6, 6), CV_64FC1, cv::Scalar(0.));
				cv::Mat Y(cv::Size(1, 6), CV_64FC1, cv::Scalar(0.));

				for (int i = 0; i < keypointsPair.size(); i++)
				{
					M.at<double>(0, 0) += (double)(std::get<1>(keypointsPair[i]).x * std::get<1>(keypointsPair[i]).x);	// a_i1^2
					M.at<double>(0, 1) += (double)(std::get<1>(keypointsPair[i]).x * std::get<1>(keypointsPair[i]).y);	// a_i1 * a_i2
					M.at<double>(0, 2) += (double)(std::get<1>(keypointsPair[i]).x);									// a_i1

					M.at<double>(1, 0) += (double)(std::get<1>(keypointsPair[i]).x * std::get<1>(keypointsPair[i]).y);	// a_i1 * a_i2
					M.at<double>(1, 1) += (double)(std::get<1>(keypointsPair[i]).y * std::get<1>(keypointsPair[i]).y);	// a_i2^2
					M.at<double>(1, 2) += (double)(std::get<1>(keypointsPair[i]).y);									// a_i2

					M.at<double>(2, 0) += (double)(std::get<1>(keypointsPair[i]).x);									// a_i1
					M.at<double>(2, 1) += (double)(std::get<1>(keypointsPair[i]).y);									// a_i2
					M.at<double>(2, 2) += (double)1.;																	// 1

					M.at<double>(3, 3) += (double)(std::get<1>(keypointsPair[i]).x * std::get<1>(keypointsPair[i]).x);	// a_i1^2
					M.at<double>(3, 4) += (double)(std::get<1>(keypointsPair[i]).x * std::get<1>(keypointsPair[i]).y);	// a_i1 * a_i2
					M.at<double>(3, 5) += (double)(std::get<1>(keypointsPair[i]).x);									// a_i1

					M.at<double>(4, 3) += (double)(std::get<1>(keypointsPair[i]).x * std::get<1>(keypointsPair[i]).y);	// a_i1 * a_i2
					M.at<double>(4, 4) += (double)(std::get<1>(keypointsPair[i]).y * std::get<1>(keypointsPair[i]).y);	// a_i2^2
					M.at<double>(4, 5) += (double)(std::get<1>(keypointsPair[i]).y);									// a_i2

					M.at<double>(5, 3) += (double)(std::get<1>(keypointsPair[i]).x);									// a_i1
					M.at<double>(5, 4) += (double)(std::get<1>(keypointsPair[i]).y);									// a_i2
					M.at<double>(5, 5) += (double)1.;																	// 1

					Y.at<double>(0, 0) += (double)(std::get<1>(keypointsPair[i]).x * std::get<0>(keypointsPair[i]).x);	// a_i1 * b_i1
					Y.at<double>(1, 0) += (double)(std::get<1>(keypointsPair[i]).y * std::get<0>(keypointsPair[i]).x);	// a_i2 * b_i1
					Y.at<double>(2, 0) += (double)(std::get<0>(keypointsPair[i]).x);									// b_i1
					Y.at<double>(3, 0) += (double)(std::get<1>(keypointsPair[i]).x * std::get<0>(keypointsPair[i]).y);	// a_i1 * b_i2
					Y.at<double>(4, 0) += (double)(std::get<1>(keypointsPair[i]).y * std::get<0>(keypointsPair[i]).y);	// a_i2 * b_i2
					Y.at<double>(5, 0) += (double)(std::get<0>(keypointsPair[i]).y);									// b_i2
				}
				

				cv::Mat M_inv = M.inv();
				cv::Mat T = M_inv*(Y);
				cv::Mat test = M * M_inv;
				cv::Mat retT = cv::Mat(cv::Size(3, 3), CV_64FC1, cv::Scalar(0.));
				
				retT.at<double>(0, 0) = T.at<double>(0, 0);
				retT.at<double>(1, 0) = T.at<double>(1, 0);
				retT.at<double>(2, 0) = T.at<double>(2, 0);
				retT.at<double>(0, 1) = T.at<double>(3, 0);
				retT.at<double>(1, 1) = T.at<double>(4, 0);
				retT.at<double>(2, 1) = T.at<double>(5, 0);
				retT.at<double>(2, 2) = 1;
				
				return retT;
			}

			double calcError(std::tuple<cv::Point2f, cv::Point2f>& pointsPair, cv::Mat& T)
			{
				double retError = 0.0;
				cv::Point2f leftPoint = std::get<0>(pointsPair);
				cv::Point2f rightPoint = std::get<1>(pointsPair);
				
				double t11, t12, t21, t22, t31, t32;	// see 식 (7.14) in page 322 

				t11 = T.at<double>(0, 0);
				t12 = T.at<double>(0, 1);
				t21 = T.at<double>(1, 0);
				t22 = T.at<double>(1, 1);
				t31 = T.at<double>(2, 0);
				t32 = T.at<double>(2, 1);

				retError = (leftPoint.x - (rightPoint.x * t11 + rightPoint.y * t21 + t31)) * (leftPoint.x - (rightPoint.x * t11 + rightPoint.y * t21 + t31))
						 + (leftPoint.y - (rightPoint.x * t12 + rightPoint.y * t22 + t32)) * (leftPoint.y - (rightPoint.x * t12 + rightPoint.y * t22 + t32));

				return retError;
			}

			cv::Mat findHomographyWithRANSAC(std::vector<cv::Point2f>& _leftKeypoints, std::vector<cv::Point2f>& _rightKeypoints)
			{
				int maxIteration = 2000;

				int inlierNumSampleBound = 10;	//same with opencv 
				double fitnessErrorBound_by_element_t = 3.0;	//same with opencv 
				double fitnessErrorBound_by_total_e = 3.0;	//same with opencv 

				double lowest_error = std::numeric_limits<float>::max();
				cv::Mat optimalTransform;

				
				for (int iter = 0; iter < 2000 ; iter++)
				{
					std::vector<cv::Point2f> leftKeypoints = _leftKeypoints;
					std::vector<cv::Point2f> rightKeypoints = _rightKeypoints;

					// Step 1.	X에서 세개의 대응점 쌍을 임의로 선택한다.
					int numKeyPoints = leftKeypoints.size();

					std::vector<std::tuple<cv::Point2f, cv::Point2f>> threeKeypointsPair;
					std::vector<std::tuple<cv::Point2f, cv::Point2f>> restKeypointsPair;
					

					srand((unsigned int)time(NULL));

					for (int i = 0; i < 3; i++)
					{
						int selectedIndex = rand() % numKeyPoints;
						threeKeypointsPair.push_back(std::make_tuple(leftKeypoints[selectedIndex], rightKeypoints[selectedIndex]));
						leftKeypoints.erase(leftKeypoints.begin() + selectedIndex);
						rightKeypoints.erase(rightKeypoints.begin() + selectedIndex);
						numKeyPoints--;
					}

					for (int i = 0; i < numKeyPoints; i++)
					{
						restKeypointsPair.push_back(std::make_tuple(leftKeypoints[i], rightKeypoints[i]));
					}

					// Step 2.	이들 세 쌍을 입력으로 식 (7.14)를 풀어 T_j 를 추정한다.
					cv::Mat curTransformWithThreePoints;
					curTransformWithThreePoints = estimateLeastSquare(threeKeypointsPair);

					// Step 3.	이들 세 쌍으로 집합 inlier를 초기화한다.
					std::vector<std::tuple<cv::Point2f, cv::Point2f>> inliers;

					// Step 4.	for(이 세 쌍을 제외한 X의 요소 p 각각에 대해)
					//			if(p가 허용오차 t 이내로 T_j에 적합)
					//				p를 inliner에 넣는다.

					for (auto& pair : restKeypointsPair)
					{
						double error;
						error = calcError(pair, curTransformWithThreePoints);
						if (error < fitnessErrorBound_by_element_t)
						{
							inliers.push_back(pair);
						}
					}

					if (inliers.empty())
						continue;
					//5.	if(inlier > d) // 집합 inliner가 d개 이상의 샘플을 가지면
					//			inlier에 있는 모든 샘플을 가지고 새로운 T를 계산한다.
					cv::Mat curTransformWithAll;
					if (inliers.size() > inlierNumSampleBound)
					{
						curTransformWithAll = estimateLeastSquare(inliers);
					}
					else
					{
						continue;
					}

					//Step 6.	if(T_j의 적합 오류 < e) T_j를 집합 Q에 넣는다.
					double fitnessError = 0.0;
					for (auto& pair : inliers)
					{
						fitnessError += calcError(pair, curTransformWithAll);
					}

					fitnessError /= inliers.size();

					if (fitnessError < fitnessErrorBound_by_total_e)
					{
						if (fitnessError < lowest_error)
						{
							lowest_error = fitnessError;
							optimalTransform = curTransformWithAll;
						}
					}
				}

				//Step 7.	Q에 있는 변환 행렬 중 가장 좋을 것을 T로 취한다. 
				// -> We just choose a transform T that has a lowest error
				return optimalTransform.t();
			}
		}
	}
}
