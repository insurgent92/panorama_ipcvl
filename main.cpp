#include <iostream>
#include "use_opencv.h"
#include "config.hpp"


void pointWiseAffineTransform(const cv::Mat &src, cv::Mat& dst, const cv::Mat T)
{
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			int x =T.at<int>(0, 0) * i + T.at<int>(0, 1) * j + T.at<int>(0, 2);
			int y =T.at<int>(1, 0) * i + T.at<int>(1, 1) * j + T.at<int>(1, 2);

			dst.at<cv::Vec3b>(y, x)[0] = src.at<cv::Vec3b>(j, i)[0];
			dst.at<cv::Vec3b>(y, x)[1] = src.at<cv::Vec3b>(j, i)[1];
			dst.at<cv::Vec3b>(y, x)[2] = src.at<cv::Vec3b>(j, i)[2];
		}
	}
}
void stitch(const cv::Mat& leftImage, const cv::Mat& rightImage, cv::OutputArray dst, const cv::Mat& rightT, int extraVerticalMargin = 30, int extraHorizontalMargin = 30)
{
	dst.create(cv::Size(leftImage.cols * 2 + extraHorizontalMargin * 2, leftImage.rows + extraVerticalMargin * 2), CV_8UC3);
	cv::Mat dstImage = dst.getMat();
	dstImage.setTo(cv::Scalar(0, 0, 0));
	cv::Mat _rightT = rightT.clone();
	

	int leftTElements[3][3] = { {1, 0, extraHorizontalMargin },{ 0, 1, extraVerticalMargin } ,{ 0, 0, 1 } };
	cv::Mat _leftT = cv::Mat(3, 3, CV_32S, &leftTElements);

	_rightT.at<int>(0, 2) += extraHorizontalMargin;
	_rightT.at<int>(1, 2) += extraVerticalMargin;

	pointWiseAffineTransform(leftImage, dstImage, _leftT);
	pointWiseAffineTransform(rightImage, dstImage, _rightT);

	//dst = dstImage.getM;
}

int main()
{
	using namespace std;
	using namespace cv;

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

	Ptr<FastFeatureDetector> fastF = FastFeatureDetector::create(5, true);
	fastF->detect(gray_src1, keypoints1);
	fastF->detect(gray_src2, keypoints2);

	cout << "keypoints1.size()=" << keypoints1.size() << endl;
	cout << "keypoints2.size()=" << keypoints2.size() << endl;

	//Step 2: Calclate descriptors
	Mat descriptor1, descriptor2;
	Ptr<xfeatures2d::BriefDescriptorExtractor> extractor = xfeatures2d::BriefDescriptorExtractor::create();
	extractor->compute(gray_src1, keypoints1, descriptor1);
	extractor->compute(gray_src2, keypoints2, descriptor2);

	//Step 3: Matching descriptor vectors
	vector<DMatch> matches;
	BFMatcher matcher(NORM_HAMMING);
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
	vector<Point2f> obj;
	vector<Point2f> scene;
	for (int i = 0; i < goodMatches.size(); i++)
	{
		//Get the keypoints from the good matches
		obj.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		scene.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	Mat H = findHomography(scene, obj,CV_RANSAC);

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

	cv::Mat dst;
	stitch(src1, src2, dst, H);

	imshow("src1", src1);
	imshow("src2", src2);
	imshow("imgMatches", imgMathes);
	imshow("dst", dst);
	waitKey();
	return 0;
}