#include <iostream>
#include <cmath>
#include "use_opencv.h"
#include "config.hpp"
#include "util.hpp"
#include "Matcher.h"
#include "HOGDescriptorExtractor.h"

int main()
{
	using namespace std;
	using namespace cv;

	const double PI = 3.141592;
	double degree = -150;

	std::cout << std::tan(degree / 180. * PI) << std::endl;
	std::cout << std::atan(-1/1.73205080757) * 180 / PI << std::endl;
	std::cout << std::atan2(-1, -1.73205080757) * 180 / PI << std::endl;
	std::cout << std::atan2(-1, 1.73205080757) * 180 / PI << std::endl;
	//return 0;

	cv::Mat src1, src2;
	cv::Mat gray_src1, gray_src2;

	src1 = imread("./sample/src1.jpg");
	src2 = imread("./sample/src2.jpg");

	if (src1.empty() || src2.empty())
		return -1;

	cvtColor(src1, gray_src1, CV_BGR2GRAY);
	cvtColor(src2, gray_src2, CV_BGR2GRAY);

	//Step 1: detect the keypoints
	vector<KeyPoint> keypoints1, keypoints2;

	Ptr<FastFeatureDetector> fastF = FastFeatureDetector::create(20, true);
	fastF->detect(gray_src1, keypoints1);
	fastF->detect(gray_src2, keypoints2);

	cout << "keypoints1.size()=" << keypoints1.size() << endl;
	cout << "keypoints2.size()=" << keypoints2.size() << endl;

	// test
	cv::Size winSize = cv::Size(32, 32);
	cv::Size blockSize = cv::Size(16, 16);
	cv::Size blockStride = cv::Size(8, 8);
	cv::Size cellSize = cv::Size(8, 8);
	int nBins = 9;
	int derivAper = 1;
	int winSigma = -1;
	int histogramNormType = 0;
	float L2HysThresh = 0.2;
	int gammaCorrection = 1;
	int n_level = 64;
	bool useSignedGradients = 1;

	cv::HOGDescriptor hog(
		winSize,
		blockSize,
		blockStride,
		cellSize,
		nBins,
		derivAper,
		winSigma,
		histogramNormType,
		L2HysThresh,
		gammaCorrection,
		n_level,
		useSignedGradients);
	//

	cv::Mat testImage;
	testImage = gray_src1(cv::Rect(keypoints1[0].pt.x - 16, keypoints1[0].pt.y - 16, 32, 32));

	std::vector<float> desctiptors;
	
	hog.compute(testImage, desctiptors);

	//Step 2: Calclate descriptors
	Mat descriptor1, descriptor2;
	
	//Ptr<xfeatures2d::BriefDescriptorExtractor> extractor = xfeatures2d::BriefDescriptorExtractor::create();

	//extractor->compute(gray_src1, keypoints1, descriptor1);
	//extractor->compute(gray_src2, keypoints2, descriptor2);

	VISIONNOOB::PANORAMA::HOGDescriptorExtractor extractor2(winSize, blockSize, blockStride, cellSize, nBins, L2HysThresh);
	extractor2.compute(gray_src1, keypoints1, descriptor1);
	extractor2.compute(gray_src2, keypoints2, descriptor2);

	//Step 3: Matching descriptor vectors
	vector<DMatch> matches;
	//BFMatcher matcher(NORM_L2);
	VISIONNOOB::PANORAMA::Matcher matcher;
	matcher.match(descriptor1, descriptor2, matches);

	cout << "matches.size()=<<" << matches.size() << endl;
	if (matches.size() < 4)
		return 0;

	//find goodMatches such that matchs[i].distance < = 4*minDist
	double minDist, maxDist;
	minDist = maxDist = matches[0].distance;

	for (int i = 1; i < matches.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist) minDist = dist;
		if (dist > maxDist) maxDist = dist;
	}

	cout << "minDist=" << minDist << endl;
	cout << "maxDist=" << maxDist << endl;

	vector<DMatch> goodMatches;
	double fTh = 2 * minDist;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance <= max(fTh, 0.02))
			goodMatches.push_back(matches[i]);
	}
	cout << "goodMatches.size()=" << goodMatches.size() << endl;
	if (goodMatches.size() < 4)
		return 0;

	//draw good matches
	Mat imgMathes;
	drawMatches(src1, keypoints1, src2, keypoints2, goodMatches, imgMathes, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Good Matches", imgMathes);

	// find Homography between keypoint1 and keypoint2
	vector<Point2f> left;
	vector<Point2f> right;
	for (int i = 0; i < goodMatches.size(); i++)
	{
		//Get the keypoints from the good matches
		left.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		right.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	//Mat H = findHomography(right, left, CV_RANSAC);
	Mat H = VISIONNOOB::PANORAMA::UTIL::findHomographyWithRANSAC(left, right);

	cv::Mat dst;
	VISIONNOOB::PANORAMA::UTIL::stitch(src1, src2, dst, H);

	vector<Point2f> objP(4);
	objP[0] = Point(0, 0);
	objP[1] = Point(gray_src1.cols, 0);
	objP[2] = Point(gray_src1.cols, gray_src1.rows);
	objP[3] = Point(0, gray_src1.rows);

	vector<Point2f> sceneP(4);
	perspectiveTransform(objP, sceneP, H);
	cout << H << endl;

	for (int i = 0; i < 4; i++)
	{
		sceneP[i] += Point2f(gray_src1.cols, 0);
	}

	for (int i = 0; i < 4; i++)
	{
		line(imgMathes, sceneP[i], sceneP[(i + 1) % 4], Scalar(255, 0, 0), 4);
	}

	imshow("src1", src1);
	imshow("src2", src2);
	imshow("imgMatches", imgMathes);
	imshow("dst", dst);
	waitKey();
	return 0;
}