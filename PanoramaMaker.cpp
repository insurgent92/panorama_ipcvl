#include "PanoramaMaker.h"

namespace visionNoob
{
	namespace computerVision
	{
		namespace apps
		{
			PanoramaMaker::PanoramaMaker()
			{
			}

			PanoramaMaker::~PanoramaMaker()
			{
			}

			void PanoramaMaker::setImages(cv::InputArray _src1, cv::InputArray _src2)
			{
				src1 = _src1.getMat();
				src2 = _src2.getMat();

				if (src1.empty() || src2.empty())
				{
					std::cout << "Can't Read files" << std::endl;
				}

				assert(!src1.empty());
				assert(!src2.empty());

				cvtColor(src1, src1_gray, CV_BGR2GRAY);
				cvtColor(src2, src2_gray, CV_BGR2GRAY);

				assert(!src1_gray.empty());
				assert(!src2_gray.empty());
			}

			void PanoramaMaker::setImages(std::string src1_path, std::string src2_path)
			{
				setImages(cv::imread(src1_path), cv::imread(src2_path));
			}

			void PanoramaMaker::getPanoramaImage(cv::OutputArray dst)
			{
				assert(!panoramaResult.empty());
				panoramaResult.copyTo(dst);
			}

			void PanoramaMaker::detectKeypoints(bool useOpenCVFunction, bool printLog)
			{
				if (useOpenCVFunction)
				{
					cv::Ptr<cv::FastFeatureDetector> fastF = cv::FastFeatureDetector::create(20, true);
					fastF->detect(src1_gray, keypoints1);
					fastF->detect(src2_gray, keypoints2);
				}
				else // else if(!useOpenCVFunction)
				{
					visionNoob::computerVision::MoravecFeatureExtractor MORAVEC;
					MORAVEC.detect(src1_gray, keypoints1, 3, 1.5E+04);
					MORAVEC.detect(src2_gray, keypoints2, 3, 1.5E+04);
				}

				if (printLog)
				{
					std::cout << "keypoints2.size()=" << keypoints2.size() << std::endl;
					std::cout << "keypoints1.size()=" << keypoints1.size() << std::endl;
				}

				assert(!keypoints1.empty());
				assert(!keypoints2.empty());
			}

			void PanoramaMaker::calcDescriptors(bool useOpenCVFunction, bool printLog)
			{
				cv::Size winSize = cv::Size(32, 32);
				cv::Size blockSize = cv::Size(16, 16);
				cv::Size blockStride = cv::Size(16, 16);
				cv::Size cellSize = cv::Size(8, 8);
				int nBins = 9;
				int derivAper = 1;
				int winSigma = -1;
				int histogramNormType = 0;
				float L2HysThresh = 0.2;
				int gammaCorrection = 1;
				int n_level = 64;
				bool useSignedGradients = 1;

				assert(!keypoints1.empty());
				assert(!keypoints2.empty());

				if (useOpenCVFunction)
				{
					cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
					extractor->compute(src1_gray, keypoints1, descriptor1);
					extractor->compute(src2_gray, keypoints2, descriptor2);
				}

				else // else if(!useOpenCVFunction)
				{
					visionNoob::computerVision::HOGDescriptorExtractor extractor2(winSize, blockSize, blockStride, cellSize, nBins, L2HysThresh);
					extractor2.compute(src1_gray, keypoints1, descriptor1);
					extractor2.compute(src2_gray, keypoints2, descriptor2);
				}

				if (printLog)
				{
					//TODO. 
				}

				assert(!descriptor1.empty());
				assert(!descriptor2.empty());
			}

			void PanoramaMaker::matchDescriptors(bool useOpenCVFunction, bool printLog)
			{
				assert(!descriptor1.empty());
				assert(!descriptor2.empty());

				if (useOpenCVFunction)
				{
					cv::BFMatcher matcher(cv::NORM_L2);
					matcher.match(descriptor1, descriptor2, matches);
				}

				else // else if(!useOpenCVFunction) 
				{
					visionNoob::computerVision::Matcher matcher;
					matcher.match(descriptor1, descriptor2, matches);
				}

				if (printLog)
				{
					std::cout << "matches.size()=<<" << matches.size() << std::endl;
				}

				assert(matches.size() >= 4);

			}

			void PanoramaMaker::findHomography(bool useOpenCVFunction, bool printLog)
			{
				std::vector<cv::Point2f> left, right;

				for (int i = 0; i < matches.size(); i++)
				{
					left.push_back(keypoints1[matches[i].queryIdx].pt);
					right.push_back(keypoints2[matches[i].trainIdx].pt);
				}
				if (useOpenCVFunction)
				{
					homography = cv::findHomography(right, left, CV_RANSAC);
				}

				else // else if(!useOpenCVFunction) 
				{
					homography = visionNoob::computerVision::util::findHomographyWithRANSAC(left, right);
				}

				assert(!homography.empty());

				if (printLog)
				{
					std::cout << "homography" << homography << std::endl;
				}

			}

			void PanoramaMaker::refineMatches(bool printLog)
			{
				const float distanceThreshold = 15;

				auto it = matches.begin();

				while (it != matches.end())
				{
					if ((*it).distance > distanceThreshold)
					{
						it = matches.erase(it);
					}

					else
					{
						++it;
					}
				}

				assert(matches.size() >= 4);

				if (printLog)
				{
					std::cout << "matches.size()=<<" << matches.size() << std::endl;
				}
			}

			void PanoramaMaker::stitchImages()
			{
				assert(!src1.empty());
				assert(!src2.empty());
				assert(!homography.empty());

				visionNoob::computerVision::util::stitch(src1, src2, panoramaResult, panoramaBinaryMask, homography);

				assert(!panoramaResult.empty());
				assert(!panoramaBinaryMask.empty());
			}

			void PanoramaMaker::getMatchingImage(cv::OutputArray dst)
			{
				cv::Mat matchingImage;

				assert(!src1.empty());
				assert(!keypoints1.empty());
				assert(!src2.empty());
				assert(!keypoints2.empty());
				assert(!matches.empty());

				drawMatches(src1, keypoints1, src2, keypoints2, matches, matchingImage, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				assert(!matchingImage.empty());
				matchingImage.copyTo(dst);
			}

			void PanoramaMaker::getPostProcessedPanoramaImage(cv::OutputArray dst)
			{
				assert(!postProcessedResult.empty());
				postProcessedResult.copyTo(dst);
			}

			void PanoramaMaker::postProcess()
			{
				//initialization
				int left_x = -1;
				int right_x = panoramaBinaryMask.cols;

				int top_y = -1;
				int bottom_y = panoramaBinaryMask.rows;

				//find top_y
				for (int idx = 0; idx < panoramaBinaryMask.cols; idx++)
				{
					for (int idy = 0; idy < panoramaBinaryMask.rows; idy++)
					{
						if (panoramaBinaryMask.at<cv::Vec3b>(idy, idx)[0] == 1)
						{
							if (idy > top_y)
							{
								top_y = idy;
							}
							break;

						}
					}
				}

				//find bottom_y
				for (int idx = 0; idx < panoramaBinaryMask.cols; idx++)
				{
					for (int idy = panoramaBinaryMask.rows - 1; idy >= 0; idy--)
					{
						if (panoramaBinaryMask.at<cv::Vec3b>(idy, idx)[0] == 1)
						{
							if (idy < bottom_y)
							{
								bottom_y = idy;
							}
							break;
						}
					}
				}

				//find left_x
				for (int idy = top_y; idy <bottom_y; idy++)
				{
					for (int idx = 0; idx < panoramaBinaryMask.cols; idx++)
					{
						if (panoramaBinaryMask.at<cv::Vec3b>(idy, idx)[0] == 1)
						{
							if (idx > left_x)
							{
								left_x = idx;
							}
							break;
						}
					}
				}

				//find right_x
				for (int idy = top_y; idy < bottom_y; idy++)
				{
					for (int idx = panoramaBinaryMask.cols - 1; idx >= 0; idx--)
					{
						if (panoramaBinaryMask.at<cv::Vec3b>(idy, idx)[0] == 1)
						{
							if (idx < right_x)
							{
								right_x = idx;
							}
							break;

						}
					}
				}

				

				cv::Mat deb = panoramaBinaryMask.clone();
				cv::Rect roi = cv::Rect(left_x, top_y, right_x - left_x, bottom_y - top_y);
				postProcessedResult = panoramaResult(roi);
			}

			void PanoramaMaker::compute()
			{
				bool useOpenCVFunction = true;
				bool printLog = true;

				//Step 1: detect the keypoints
				detectKeypoints(!useOpenCVFunction, printLog);

				//Step 2: calculate descriptors
				calcDescriptors(!useOpenCVFunction, printLog);

				//Step 3: match descriptor vectors
				matchDescriptors(!useOpenCVFunction, printLog);

				//Step 4: find good matches
				refineMatches(printLog);

				//Step 5: find homography between keypoint1 and keypoint2
				findHomography(!useOpenCVFunction, printLog);

				//Step 6 : stitch together based on Homograpy
				stitchImages();

				//Step 7 : postProcess
				postProcess();
			}
		}
	}
}


