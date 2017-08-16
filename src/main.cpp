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
#include "Stixels.h"
#include "RoadEstimation.h"

void HSV_to_RGB(const float h, const float s, const float v, int *cr, int *cg, int *cb) {
    const float h_prima = h*360.0f / 60.0f;

    const float c = v*s;
    const float h_mod = fmodf(h_prima, 2.0f);
    const float x = c*(1.0f - fabsf(h_mod - 1.0f));

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    if (h_prima >= 0) {
        if (h_prima < 1.0f) {
            r = c;
            g = x;
            b = 0.0f;
        }
        else if (h_prima < 2.0f) {
            r = x;
            g = c;
            b = 0.0f;
        }
        else if (h_prima < 3.0f) {
            r = 0.0f;
            g = c;
            b = x;
        }
        else if (h_prima < 4.0f) {
            r = 0.0f;
            g = x;
            b = c;
        }
        else if (h_prima < 5.0f) {
            r = x;
            g = 0.0f;
            b = c;
        }
        else if (h_prima < 6.0f) {
            r = c;
            g = 0.0f;
            b = x;
        }
    }

    const float m = v - c;
    r = (r + m)*255.0f;
    g = (g + m)*255.0f;
    b = (b + m)*255.0f;
    *cr = (int)r;
    *cg = (int)g;
    *cb = (int)b;

}

cv::Mat colorTable;
bool colorTableInitialized = false;

int main(int argc, char *argv[]) {
	if(argc < 3) {
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

	const char* left_video_path = argv[1];
    const char* right_video_path = argv[2];

    std::cout << left_video_path << std::endl;

    cv::VideoCapture left_video(left_video_path);
    cv::VideoCapture right_video(right_video_path);
    if (!left_video.isOpened() || !right_video.isOpened()) {
        std::cerr << "open video file failed" << std::endl;
    }

    bool first_time = true;
    DisparityEstimation disparity_estimation;
    Stixels stixles;
    RoadEstimation road_estimation;
    disparity_estimation.Initialize();
    road_estimation.Initialize();
    stixles.Initialize();
    disparity_estimation.SetParameter(p1, p2);
    road_estimation.SetCameraParameters(camera_parameters);
    stixles.SetParameters(probabilities_parameters, camera_parameters, disparity_parameters, stixel_model_parameters);

    cv::Mat left_frame, right_frame, left_frame1, right_frame1;
    cv::Mat disparity_im, disparity_im_color;
    cv::Size frame_size = cv::Size(0, 0);
    cv::Size rect_size = cv::Size(0, 0);
    cv::Rect rect_roi_up = cv::Rect(0, 0, 0, 0);
    cv::Rect rect_roi_down = cv::Rect(0, 0, 0, 0);

    cv::Mat mix_frame;
    cv::namedWindow("Test");

    int currentFrame = 0;
    while (true)
    {
        if (!left_video.read(left_frame) || !right_video.read(right_frame))
        {
            std::cerr << "Reach the end of video file" << std::endl;
            left_video.set(CV_CAP_PROP_POS_FRAMES, 0);
            right_video.set(CV_CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        currentFrame++;
        if (left_frame.size() != right_frame.size()) {
            std::cerr << "Both images must have the same dimensions" << std::endl;
            return EXIT_FAILURE;
        }

        if (left_frame.size() != frame_size)
        {
            frame_size = left_frame.size();
            std::cout << "FrameSize:" << frame_size.height << "*" << frame_size.width << std::endl;
            cv::Size newSize = cv::Size(frame_size.width / 4 * 4, frame_size.height / 4 * 4);
            if (newSize != rect_size)
            {
                rect_size = newSize;
                rect_roi_up = cv::Rect(0, 0, rect_size.width, rect_size.height);
                rect_roi_down = cv::Rect(0, rect_size.height, rect_size.width, rect_size.height);
                std::cout << "CutSize:" << rect_size.height << "*" << rect_size.width << std::endl;
                mix_frame = cv::Mat::zeros(cv::Size(rect_size.width, rect_size.height * 2), CV_8UC3);
                road_estimation.UpdateImageSize(rect_size);
                stixles.UpdateImageSize(rect_size);
                disparity_estimation.UpdateImageSize(rect_size);
            }
        }

        left_frame = left_frame(rect_roi_up);
        right_frame = right_frame(rect_roi_up);
        
        if (left_frame.channels() > 1) {
            cv::cvtColor(left_frame, left_frame1, CV_RGB2GRAY);
        }
            
        if (right_frame.channels() > 1) {
            cv::cvtColor(right_frame, right_frame1, CV_RGB2GRAY);
        }
        disparity_estimation.LoadImages(left_frame1, right_frame1);
        float elapsed_time_ms;
        disparity_estimation.Compute(&elapsed_time_ms);
        std::cout << currentFrame << ":" << elapsed_time_ms << "ms" << std::endl;
        disparity_im = disparity_estimation.FetchDisparityResult();
        disparity_im_color = disparity_estimation.FetchColoredDisparityResult();
        pixel_t* disparityResultPixelD = disparity_estimation.FetchDisparityResultPixelD();
        road_estimation.LoadDisparityImageD(disparityResultPixelD);
        const bool ok = road_estimation.Compute();
        if (!ok) {
            printf("Can't compute road estimation\n");
            continue;
        }

        // Get Camera Parameters
        estimated_camera_parameters = road_estimation.FetchEstimatedCameraParameters();

        if (estimated_camera_parameters.pitch == 0 && estimated_camera_parameters.cameraHeight == 0 && estimated_camera_parameters.horizonPoint == 0 && estimated_camera_parameters.slope == 0) {
            printf("Can't compute road estimation\n");
            continue;
        }

        std::cout << "Camera Parameters -> Tilt: " << estimated_camera_parameters.pitch << " Height: " << estimated_camera_parameters.cameraHeight << " vHor: " << estimated_camera_parameters.horizonPoint << " alpha_ground: " << estimated_camera_parameters.slope << std::endl;

        if (estimated_camera_parameters.cameraHeight < 0 || estimated_camera_parameters.cameraHeight > 5 ||
            estimated_camera_parameters.horizonPoint < 0 || estimated_camera_parameters.slope < 0) {
            printf("Error happened in computing road estimation\n");
            continue;
        }

        stixles.LoadDisparityImageD(disparityResultPixelD, estimated_camera_parameters);
        stixles.Compute(&elapsed_time_ms);
        Section *stx = stixles.FetchStixels();

        cv::Mat left_frame_stx;
        left_frame.copyTo(left_frame_stx);

        std::vector< std::vector<Section> > stixels;
        stixels.resize(stixles.FetchRealCols());

        for (size_t i = 0; i < stixels.size(); i++) {
            for (size_t j = 0; j < stixel_model_parameters.maxSections; j++) {
                Section section = stx[i*stixel_model_parameters.maxSections + j];
                if (section.type == -1) {
                    break;
                }
                // If disparity is 0 it is sky
                if (section.type == OBJECT && section.disparity < 1.0f) {
                    section.type = SKY;
                }
                stixels[i].push_back(section);
            }
            // Column finished
        }

        for (size_t i = 0; i < stixels.size(); i++) {
            std::vector<Section> column = stixels.at(i);
            Section prev;
            prev.type = -1;
            bool have_prev = false;
            for (size_t j = 0; j < column.size(); j++) {
                Section sec = column.at(j);
                sec.vB = left_frame_stx.rows - 1 - sec.vB;
                sec.vT = left_frame_stx.rows - 1 - sec.vT;

                // If disparity is 0 it is sky
                if (sec.type == OBJECT && sec.disparity < 1.0f) {
                    sec.type = SKY;
                }

                // Sky on top of sky
                if (j > 0) {
                    if (!have_prev) {
                        prev = column.at(j - 1);
                        prev.vB = left_frame_stx.rows - 1 - prev.vB;
                        prev.vT = left_frame_stx.rows - 1 - prev.vT;
                    }

                    if (sec.type == SKY && prev.type == SKY) {
                        sec.vT = prev.vT;
                        have_prev = true;
                    }
                    else {
                        have_prev = false;
                    }
                }

                // If the next segment is a sky, skip current
                if (j + 1 < column.size() && sec.type == SKY && column.at(j + 1).type == SKY) {
                    continue;
                }
                // Don't show ground
                if (sec.type != GROUND) {
                    const int x = i*stixel_model_parameters.columnStep + stixel_model_parameters.widthMargin;
                    const int y = sec.vT;
                    const int width = stixel_model_parameters.columnStep;
                    int height = sec.vB - sec.vT + 1;

                    cv::Mat roi = left_frame_stx(cv::Rect(x, y, width, height));

                    // Sky = blue
                    int cr = 0;
                    int cg = 0;
                    int cb = 255;

                    // Object = from green to red (far to near)
                    if (sec.type == OBJECT) {
                        const float dis = (max_dis_display - sec.disparity) / max_dis_display;
                        float dis_f = dis;
                        if (dis_f < 0.0f) {
                            dis_f = 0.0f;
                        }
                        const float h = dis_f*0.3f;
                        const float s = 1.0f;
                        const float v = 1.0f;

                        HSV_to_RGB(h, s, v, &cr, &cg, &cb);
                    }

                    cv::Mat color;
                    const int top = (roi.rows < 2) ? 0 : 1;
                    const int bottom = (roi.rows < 2) ? 0 : 1;
                    const int left = 1;
                    const int right = 1;

                    color.create(roi.rows - top - bottom, roi.cols - left - right, roi.type());
                    color.setTo(cv::Scalar(cb, cg, cr));

                    cv::Mat color_padded;
                    color_padded.create(roi.rows, roi.cols, color.type());

                    copyMakeBorder(color, color_padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
                    const double alpha = 0.6;
                    cv::addWeighted(color_padded, alpha, roi, 1.0 - alpha, 0.0, roi);

                }
            }
        }
        if (estimated_camera_parameters.horizonPoint != -1) {
            // Draw Horizon Line
            int thickness = 2;
            int lineType = 8;
            line(left_frame_stx,
                cv::Point(0, estimated_camera_parameters.horizonPoint),
                cv::Point(left_frame_stx.cols - 1, estimated_camera_parameters.horizonPoint),
                cv::Scalar(0, 0, 0),
                thickness,
                lineType);
        }

        //MIX IMGS TOGETHER
        //left_frame.copyTo(mix_frame(rect_roi_up));
        //cv::cvtColor(disparity_im, disparity_im_color, CV_GRAY2BGR);
        left_frame_stx.copyTo(mix_frame(rect_roi_up));
        disparity_im_color.copyTo(mix_frame(rect_roi_down));
        cv::imshow("Test", mix_frame);

        char c = cv::waitKey(100);
        if (c == 27) break;
        if (c != -1)
        {
            std::cout << "PAUSED" << c << std::endl;
            cv::waitKey(0);
        }

    }

    stixles.Finish();
    road_estimation.Finish();
    disparity_estimation.Finish();
	return 0;
}
