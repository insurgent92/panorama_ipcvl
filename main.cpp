#include <iostream>
#include "PanoramaMaker.h"

int main()
{
	using namespace std;
	using namespace cv;

	visionNoob::computerVision::apps::PanoramaMaker pm;
	cv::Mat panoramaImage;
	cv::Mat postProcessed;
	cv::Mat matchingImage;
	
	pm.setImages("./sample/src1.jpg", "./sample/src2.jpg");
	pm.compute();
	pm.getPanoramaImage(panoramaImage);
	pm.getMatchingImage(matchingImage);
	pm.getPostProcessedPanoramaImage(postProcessed);

	imshow("panoramaImage", panoramaImage);
	imshow("matchingImage", matchingImage);
	imshow("postProcessed", postProcessed);

	waitKey();
	return 0;
}