#include "HOGDescriptorExtractor.h"

namespace VISIONNOOB
{
	namespace PANORAMA
	{
		HOGDescriptorExtractor::HOGDescriptorExtractor(
			cv::Size _winSize,
			cv::Size _blockSize,
			cv::Size _blockStride,
			cv::Size _cellSize,
			int _nBins,
			float _L2HysThresh)
		{
			winSize = _winSize;
			blockSize = _blockSize;
			blockStride = _blockStride;
			cellSize = _cellSize;
			nBins = _nBins;
			L2HysThresh = _L2HysThresh;
		}


		HOGDescriptorExtractor::~HOGDescriptorExtractor()
		{
		}

		void HOGDescriptorExtractor::refineKeypoints(std::vector<cv::KeyPoint>& keypoints, int cols, int rows)
		{
			auto it = keypoints.begin();

			while (it != keypoints.end()) {

				int roi_x = (*it).pt.x - winSize.width / 2;
				int roi_y = (*it).pt.y - winSize.height / 2;
				int roi_width = winSize.width;
				int roi_height = winSize.height;

				if ((roi_x < 0) || (roi_y < 0) || (roi_x + roi_width >= cols) || (roi_y + roi_height >= rows))
				{
					it = keypoints.erase(it);
					std::cout << "skip this keypoint" << std::endl;
				}
				else ++it;
			}
		}

		//see http://blog.naver.com/PostView.nhn?blogId=tommybee&logNo=221173056260&parentCategoryNo=&categoryNo=57&viewDate=&isShowPopularPosts=true&from=search
		void HOGDescriptorExtractor::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray _descriptors)
		{
			cv::Mat inputImage = image.getMat();
			std::tuple<cv::Mat, cv::Mat> derivatives;
			std::tuple<cv::Mat, cv::Mat> degreeAndMagnitude;
			VISIONNOOB::PANORAMA::UTIL::calcSobel(image, derivatives);
			VISIONNOOB::PANORAMA::UTIL::calcGradientAndMagnitute(derivatives, degreeAndMagnitude);

			cv::Mat degree = std::get<0>(degreeAndMagnitude);
			cv::Mat manitude = std::get<1>(degreeAndMagnitude);

			int blockTotalIter = (winSize.width - blockSize.width) / blockStride.width + 1;
			int numCellsPerBlock = blockSize.width / cellSize.width;
			int dimVector = nBins * blockTotalIter * blockTotalIter * numCellsPerBlock * numCellsPerBlock;

			refineKeypoints(keypoints, inputImage.cols, inputImage.rows);

			cv::Mat descriptors(cv::Size(dimVector, keypoints.size()), CV_64FC1, cv::Scalar(0));


			for (int keyPointIter = 0; keyPointIter < keypoints.size(); keyPointIter++)
			{
				int roi_x = keypoints[keyPointIter].pt.x - winSize.width / 2;
				int roi_y = keypoints[keyPointIter].pt.y - winSize.height / 2;
				int roi_width = winSize.width;
				int roi_height = winSize.height;

				cv::Mat currentROI = inputImage(cv::Rect(roi_x, roi_y, roi_width, roi_height));
				cv::Mat currentROI_Mag = manitude(cv::Rect(roi_x, roi_y, roi_width, roi_height));
				cv::Mat currentROI_Deg = degree(cv::Rect(roi_x, roi_y, roi_width, roi_height));

				for (int blockIter_y = 0; blockIter_y < blockTotalIter; blockIter_y++)
				{
					for (int blockIter_x = 0; blockIter_x < blockTotalIter; blockIter_x++)
					{
						std::vector<std::vector<double>> block_hists;
						std::vector<double> current_hist;
						current_hist.resize(nBins);

						int block_roi_x = 0 + blockStride.width * blockIter_x;
						int block_roi_y = 0 + blockStride.height * blockIter_y;
						int block_roi_w = blockSize.width;
						int block_roi_h = blockSize.height;

						cv::Mat currentBlock = currentROI(cv::Rect(block_roi_x, block_roi_y, block_roi_w, block_roi_h));
						cv::Mat currentMag = currentROI_Mag(cv::Rect(block_roi_x, block_roi_y, block_roi_w, block_roi_h));
						cv::Mat currentDeg = currentROI_Deg(cv::Rect(block_roi_x, block_roi_y, block_roi_w, block_roi_h));

						std::vector<cv::Mat> cells;
						for (int cellIter_y = 0; cellIter_y < numCellsPerBlock; cellIter_y++)
						{
							for (int cellIter_x = 0; cellIter_x < numCellsPerBlock; cellIter_x++)
							{
								int cell_roi_x = 0 + cellSize.width * cellIter_x;
								int cell_roi_y = 0 + cellSize.height * cellIter_y;
								int cell_roi_w = cellSize.width;
								int cell_roi_h = cellSize.height;

								cv::Mat currentCell = currentBlock(cv::Rect(cell_roi_x, cell_roi_y, cell_roi_w, cell_roi_h));
								cv::Mat cellMeg = currentMag(cv::Rect(cell_roi_x, cell_roi_y, cell_roi_w, cell_roi_h));
								cv::Mat cellDegree = currentDeg(cv::Rect(cell_roi_x, cell_roi_y, cell_roi_w, cell_roi_h));

								for (int cell_y = 0; cell_y < currentCell.rows; cell_y++)
								{
									for (int cell_x = 0; cell_x < currentCell.cols; cell_x++)
									{
										double cur_degree = cellDegree.at<double>(cell_y, cell_x);

										if (cur_degree >= 0 && cur_degree < 40)
										{
											current_hist[0] += cellMeg.at<double>(cell_y, cell_x);
										}

										else if (cur_degree >= 40 && cur_degree < 80)
										{
											current_hist[1] += cellMeg.at<double>(cell_y, cell_x);
										}

										else if (cur_degree >= 80 && cur_degree < 120)
										{
											current_hist[2] += cellMeg.at<double>(cell_y, cell_x);
										}

										else if (cur_degree >= 120 && cur_degree < 160)
										{
											current_hist[3] += cellMeg.at<double>(cell_y, cell_x);
										}

										else if (cur_degree >= 160 && cur_degree < 200)
										{
											current_hist[4] += cellMeg.at<double>(cell_y, cell_x);
										}

										else if (cur_degree >= 200 && cur_degree < 240)
										{
											current_hist[5] += cellMeg.at<double>(cell_y, cell_x);
										}

										else if (cur_degree >= 240 && cur_degree < 280)
										{
											current_hist[6] += cellMeg.at<double>(cell_y, cell_x);
										}

										else if (cur_degree >= 280 && cur_degree < 320)
										{
											current_hist[7] += cellMeg.at<double>(cell_y, cell_x);
										}

										else if (cur_degree >= 320 && cur_degree < 360)
										{
											current_hist[8] += cellMeg.at<double>(cell_y, cell_x);
										}

									}
								}
								block_hists.push_back(current_hist);
							}
						}

						//normalize
						double magnitude = 0.;

						for (int hist_i = 0; hist_i < block_hists.size(); hist_i++)
						{
							for (int element = 0; element < nBins; element++)
							{
								magnitude += block_hists[hist_i][element];
							}
						}

						magnitude = cv::sqrt(magnitude);

						for (int hist_i = 0; hist_i < block_hists.size(); hist_i++)
						{
							for (int element = 0; element < nBins; element++)
							{
								block_hists[hist_i][element] /= magnitude;
							}
						}

						int block_idx = blockTotalIter * blockIter_y + blockIter_x;

						for (int hist_i = 0; hist_i < block_hists.size(); hist_i++)
						{
							cv::Mat featureVector(cv::Size(nBins, 1), CV_64FC1);
							memcpy(featureVector.data, block_hists[hist_i].data(), block_hists[hist_i].size() * sizeof(double));

							cv::Mat roi = descriptors(cv::Rect(block_idx * nBins *  numCellsPerBlock * numCellsPerBlock + nBins * hist_i, keyPointIter, nBins, 1));
							featureVector.copyTo(roi);
						}
					}
				}


			}

			descriptors.copyTo(_descriptors);
		}
	}
}
