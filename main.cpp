#include <iostream>
#include <cmath>
#include "use_opencv.h"
#include "config.hpp"
#include "util.hpp"
#include "Matcher.h"
#include "HOGDescriptorExtractor.h"
#include "FeatureExtractor.h"

int main()
{
	using namespace std;
	using namespace cv;

	///////////////////////////
	/*Step 0: read two images*/
	///////////////////////////
	cv::Mat src1, src2;
	cv::Mat gray_src1, gray_src2;

	src1 = imread("./sample/src1.jpg");
	src2 = imread("./sample/src2.jpg");

	if (src1.empty() || src2.empty())
		return -1;

	cvtColor(src1, gray_src1, CV_BGR2GRAY);
	cvtColor(src2, gray_src2, CV_BGR2GRAY);

	////////////////////////////////
	/*Step 1: detect the keypoints*/
	////////////////////////////////
	vector<KeyPoint> keypoints1, keypoints2;

	//Ptr<FastFeatureDetector> fastF = FastFeatureDetector::create(20, true);
	//fastF->detect(gray_src1, keypoints1);
	//fastF->detect(gray_src2, keypoints2);

	VISIONNOOB::PANORAMA::FeatureExtractor MORAVEC;
	MORAVEC.detect(gray_src1, keypoints1, 3, 1.5E+04);
	MORAVEC.detect(gray_src2, keypoints2, 3, 1.5E+04);

	cout << "keypoints1.size()=" << keypoints1.size() << endl;
	cout << "keypoints2.size()=" << keypoints2.size() << endl;

	////////////////////////////////
	/*Step 2: Calclate descriptors*/
	////////////////////////////////
	Mat descriptor1, descriptor2;
	
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

	//Ptr<xfeatures2d::BriefDescriptorExtractor> extractor = xfeatures2d::BriefDescriptorExtractor::create();
	//extractor->compute(gray_src1, keypoints1, descriptor1);
	//extractor->compute(gray_src2, keypoints2, descriptor2);

	VISIONNOOB::PANORAMA::HOGDescriptorExtractor extractor2(winSize, blockSize, blockStride, cellSize, nBins, L2HysThresh);
	extractor2.compute(gray_src1, keypoints1, descriptor1);
	extractor2.compute(gray_src2, keypoints2, descriptor2);

	///////////////////////////////////////
	/*Step 3: Matching descriptor vectors*/
	///////////////////////////////////////
	vector<DMatch> matches;

	//BFMatcher matcher(NORM_L2);
	VISIONNOOB::PANORAMA::Matcher matcher;
	matcher.match(descriptor1, descriptor2, matches);

	cout << "matches.size()=<<" << matches.size() << endl;
	if (matches.size() < 4)
		return 0;

	///////////////////////////////////////////////////////////////////////
	/*Step 4: find goodMatches such that matchs[i].distance < = 4*minDist*/
	///////////////////////////////////////////////////////////////////////
	vector<DMatch> goodMatches;
	double minDist, maxDist;

	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance <= 15)
			goodMatches.push_back(matches[i]);
	}

	cout << "goodMatches.size()=" << goodMatches.size() << endl;

	if (goodMatches.size() < 4)
		return 0;

	///////////////////////////////////////////////////////////
	/*Step 5: find Homography between keypoint1 and keypoint2*/
	///////////////////////////////////////////////////////////
	Mat H;
	vector<Point2f> left, right;

	for (int i = 0; i < goodMatches.size(); i++)
	{
		//Get the keypoints from the good matches
		left.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		right.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	//Mat H = findHomography(right, left, CV_RANSAC);
	H = VISIONNOOB::PANORAMA::UTIL::findHomographyWithRANSAC(left, right);

	//////////////////////////////////////////////
	/*Step 6: stitch together based on Homograpy*/
	//////////////////////////////////////////////
	cv::Mat dst;
	VISIONNOOB::PANORAMA::UTIL::stitch(src1, src2, dst, H);
	
	////////////////////////
	/*Step 7: show results*/
	////////////////////////
	Mat imgMathes;

	//draw good matches
	drawMatches(src1, keypoints1, src2, keypoints2, goodMatches, imgMathes, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("src1", src1);
	imshow("src2", src2);
	imshow("imgMatches", imgMathes);
	imshow("dst", dst);

	waitKey();
	return 0;
}