/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/



#include <iostream>
#ifdef WIN32
#include <Windows.h>
#include <shlwapi.h>
#include <windef.h>
#define PATH_MAX MAX_PATH 
#else
#include <sys/time.h>
#include <linux/limits.h>
#endif

#include <dirent.h>
#include <numeric>
#include <vector>
#include <stdlib.h>
#include <typeinfo>
#include <opencv2/opencv.hpp>

#include <numeric>
#include <stdlib.h>
#include <stdint.h>
#include <ctime>
#include <sys/types.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include "configuration.h"
#include "configuration_parameters.h"
#include "debug.h"
#include "DisparityEstimation.h"

int main(int argc, char *argv[]) {
	if(argc < 2) {
		std::cerr << "Usage: sgm left_video right_video" << std::endl;
		return EXIT_FAILURE;
	}
	if(MAX_DISPARITY != 128) {
		std::cerr << "Due to implementation limitations MAX_DISPARITY must be 128" << std::endl;
		return EXIT_FAILURE;
	}
	if(PATH_AGGREGATION != 4 && PATH_AGGREGATION != 8) {
        std::cerr << "Due to implementation limitations PATH_AGGREGATION must be 4 or 8" << std::endl;
        return EXIT_FAILURE;
    }

	const char* left_video_path = argv[0];
    const char* right_video_path = argv[1];

    cv::VideoCapture left_video(left_video_path);
    cv::VideoCapture right_video(right_video_path);
    if (!left_video.isOpened() || !right_video.isOpened()) {
        std::cerr << "open video file failed" << std::endl;
    }

    DisparityEstimation disparity_estimation;
    disparity_estimation.Initialize(p1, p2);
    cv::Mat left_frame, right_frame, left_frame1, right_frame1;
    cv::Mat disparity_im, disparity_im_color;
    cv::Size frame_size = cv::Size(0, 0);
    cv::Size rect_size = cv::Size(0, 0);
    cv::Rect rect_roi = cv::Rect(0, 0, 0, 0);
    cv::Rect rect_roi2 = cv::Rect(0, 0, 0, 0);

    cv::Mat mix_frame;

    cv::namedWindow("Test");

    int currentFrame = 0;
    while (true)
    {
        if (!left_video.read(left_frame) || !right_video.read(right_frame))
        {
            std::cerr << "Reach the end of video file" << std::endl;
            return EXIT_SUCCESS;
        }
        if (left_frame.size() != right_frame.size()) {
            std::cerr << "Both images must have the same dimensions" << std::endl;
            return EXIT_FAILURE;
        }       

        if (left_frame.size() != frame_size)
        {
            frame_size = left_frame.size();
            rect_size = cv::Size(frame_size.width / 4 * 4, frame_size.height / 4 * 4);
            rect_roi = cv::Rect(0, 0, rect_size.width, rect_size.height);
            rect_roi2 = cv::Rect(0, rect_size.height, rect_size.width, rect_size.height);

            std::cout << "FrameSize:" << frame_size.height << "*" << frame_size.width << std::endl;
            std::cout << "CutSize:" << rect_size.height << "*" << rect_size.width << std::endl;

            mix_frame = cv::Mat::zeros(cv::Size(rect_size.width, rect_size.height * 2), CV_8UC3);
        }

        left_frame = left_frame(rect_roi);
        right_frame = right_frame(rect_roi);
        
        if (left_frame.channels() > 1) {
            cv::cvtColor(left_frame, left_frame1, CV_RGB2GRAY);
        }
            
        if (right_frame.channels() > 1) {
            cv::cvtColor(right_frame, right_frame1, CV_RGB2GRAY);
        }

        
        float elapsed_time_ms;
        disparity_im = disparity_estimation.Compute(left_frame1, right_frame1, &elapsed_time_ms);
        std::cout << currentFrame << ":" << elapsed_time_ms << "ms" << std::endl;

        std::cout << disparity_im.type() << std::endl;

        left_frame.copyTo(mix_frame(rect_roi));
        cv::cvtColor(disparity_im, disparity_im_color, CV_GRAY2BGR);
        //right_frame.copyTo(mix_frame(rect_roi2));
        disparity_im_color.copyTo(mix_frame(rect_roi2));
        cv::imshow("Test", mix_frame);

        //if (h_im0.rows % 4 != 0 || h_im0.cols % 4 != 0) {
        //    std::cerr << "Due to implementation limitations image width and height must be a divisible by 4" << std::endl;
        //    return EXIT_FAILURE;
        //}
        int c = cv::waitKey(100);
        if ((char)c == 27) break;
        if (c > 0) cv::waitKey(0);
        currentFrame++;
    }

//	while ((ep = readdir(dp)) != NULL) {
//		// Skip directories
//		if (!strcmp (ep->d_name, "."))
//			continue;
//		if (!strcmp (ep->d_name, ".."))
//			continue;
//
//		sprintf(left_file, "%s/%s/%s", directory, left_dir, ep->d_name);
//		sprintf(right_file, "%s/%s/%s", directory, right_dir, ep->d_name);
//		sprintf(dis_file, "%s/%s/%s", directory, disparity_dir, ep->d_name);
//		sprintf(gt_file, "%s/%s/%s", directory, gt_dir, ep->d_name);
//		size_t gt_len = strlen(gt_file);
//
//		cv::Mat h_im0 = cv::imread(left_file);
//		if(!h_im0.data) {
//			std::cerr << "Couldn't read the file " << left_file << std::endl;
//			return EXIT_FAILURE;
//		}
//		cv::Mat h_im1 = cv::imread(right_file);
//		if(!h_im1.data) {
//			std::cerr << "Couldn't read the file " << right_file << std::endl;
//			return EXIT_FAILURE;
//		}
//
//		// Convert images to grayscale
//		if (h_im0.channels()>1) {
//			cv::cvtColor(h_im0, h_im0, CV_RGB2GRAY);
//		}
//
//		if (h_im1.channels()>1) {
//			cv::cvtColor(h_im1, h_im1, CV_RGB2GRAY);
//		}
//
//		
//
//#if LOG
//		std::cout << "processing: " << left_file << std::endl;
//#endif
//		// Compute
//		float elapsed_time_ms;
//		cv::Mat disparity_im = disparity_estimation.Compute(h_im0, h_im1, &elapsed_time_ms);
//#if LOG
//		std::cout << "done" << std::endl;
//#endif
//		times.push_back(elapsed_time_ms);
//
//		if(has_gt) {
//			disparity_errors(disparity_im, gt_file, &n, &n_err);
//		}
//#if WRITE_FILES
//	const int type = disparity_im.type();
//	const uchar depth = type & CV_MAT_DEPTH_MASK;
//	if(depth == CV_8U) {
//		cv::imwrite(dis_file, disparity_im);
//	} else {
//		cv::Mat disparity16(disparity_im.rows, disparity_im.cols, CV_16UC1);
//		for(int i = 0; i < disparity_im.rows; i++) {
//			for(int j = 0; j < disparity_im.cols; j++) {
//				const float d = disparity_im.at<float>(i, j)*256.0f;
//				disparity16.at<uint16_t>(i, j) = (uint16_t) d;
//			}
//		}
//		cv::imwrite(dis_file, disparity16);
//	}
//#endif
//	}
//	closedir(dp);
//    disparity_estimation.Finish();
//
//	double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
//	if(has_gt) {
//		printf("%f\n", (float) n_err/n);
//        std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f / mean << " fps" << std::endl;
//	} else {
//		std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f/mean << " fps" << std::endl;
//	}
//
	return 0;
}
