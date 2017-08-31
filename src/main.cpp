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
#include <opencv2/highgui.hpp>  

#include <numeric>
#include <stdlib.h>
#include <stdint.h>
#include <ctime>
#include <sys/types.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <fstream>
#include "configuration.h"
#include "debug.h"
#include "DisparityEstimation.h"
#include "ConfigurationGroup.h"
#include "Stixels.h"
#include "RoadEstimation.h"
#include "showStixels.h"

int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cerr << "Usage: sgm config_file" << std::endl;
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

    configurationGroup config;
    std::ifstream config_file(argv[1]);
    config_file >> config;
    config_file.close();

    const std::string left_video_path = config["left_video_path"];
    const std::string right_video_path = config["right_video_path"];
    const float prescale = CFGDOUBLE("prescale");
    const bool need_rect = CFGBOOL("need_rect");
    const int video_alignment = CFGDOUBLE("video_alignment");

    const cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) <<
        CFGDOUBLE("l_fx") * prescale, 0, CFGDOUBLE("l_cx") * prescale,
        0, CFGDOUBLE("l_fy") * prescale, CFGDOUBLE("l_cy") * prescale,
        0, 0, 1);
    const cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) <<
        CFGDOUBLE("r_fx") * prescale, 0, CFGDOUBLE("r_cx") * prescale,
        0, CFGDOUBLE("r_fy") * prescale, CFGDOUBLE("r_cy") * prescale,
        0, 0, 1);
    const CameraParameters camera_parameters = {
        /*cameraCenterX      = */ CFGDOUBLE("l_cx") * prescale,
        /*cameraCenterY      = */ CFGDOUBLE("l_cy") * prescale,
        /*baseline           = */ CFGDOUBLE("tr_x"),
        /*focal              = */ CFGDOUBLE("l_fx")
    };
    const cv::Mat cameraDistCoeffL = (cv::Mat_<double>(1, 5) <<
        CFGDOUBLE("l_k1"), CFGDOUBLE("l_k2"),
        CFGDOUBLE("l_p1"), CFGDOUBLE("l_p2"),
        CFGDOUBLE("l_k3"));
    const cv::Mat cameraDistCoeffR = (cv::Mat_<double>(1, 5) <<
        CFGDOUBLE("r_k1"), CFGDOUBLE("r_k2"),
        CFGDOUBLE("r_p1"), CFGDOUBLE("r_p2"),
        CFGDOUBLE("r_k3"));

    const cv::Mat cameraRotationMatrix = (cv::Mat_<double>(3, 3) <<
        CFGDOUBLE("rt_11"), CFGDOUBLE("rt_12"), CFGDOUBLE("rt_13"),
        CFGDOUBLE("rt_21"), CFGDOUBLE("rt_22"), CFGDOUBLE("rt_23"),
        CFGDOUBLE("rt_31"), CFGDOUBLE("rt_32"), CFGDOUBLE("rt_33"));

    const cv::Mat cameraTranslationMatrix = (cv::Mat_<double>(3, 1) <<
        CFGDOUBLE("tr_x"),
        CFGDOUBLE("tr_y"),
        CFGDOUBLE("tr_z"));

    const int p1 = CFGDOUBLE("p1");
    const int p2 = CFGDOUBLE("p2");

    const RoadEstimationParameters road_estimation_parameters = {
        /* rangeAngleX       = */ CFGDOUBLE("range_angle_x"),
        /* rangeAngleY       = */ CFGDOUBLE("range_angle_y"),
        /* houghAccumThr     = */ CFGDOUBLE("hough_accum_thr"),
        /* binThr            = */ CFGDOUBLE("bin_thr"),
        /* maxPitch          = */ CFGDOUBLE("max_pitch") DEGREE,
        /* minPitch          = */ CFGDOUBLE("min_pitch") DEGREE,
        /* maxCameraHeight   = */ CFGDOUBLE("max_cam_height"),
        /* minCameraHeight   = */ CFGDOUBLE("min_cam_height"),
    };

    const ProbabilitiesParameters probabilities_parameters = {
        /* out               = */ CFGDOUBLE("out"),
        /* outSky            = */ CFGDOUBLE("out_sky"),
        /* groundGivenNExist = */ CFGDOUBLE("ground_gne"),
        /* objectGivenNExist = */ CFGDOUBLE("object_gne"),
        /* skyGivenNExist    = */ CFGDOUBLE("sky_gne"),
        /* nExistDis         = */ CFGDOUBLE("ned"),
        /* ground            = */ CFGDOUBLE("ground"),
        /* object            = */ CFGDOUBLE("object"),
        /* sky               = */ CFGDOUBLE("sky"),
        /* ord               = */ CFGDOUBLE("ord"),
        /* grav              = */ CFGDOUBLE("grav"),
        /* blg               = */ CFGDOUBLE("blg")
    };

    const StixelModelParameters stixel_model_parameters = {
        /* columnStep        = */ CFGDOUBLE("column_step"),
        /* medianStep        = */ CFGBOOL("median_step"),
        /* epsilon           = */ CFGDOUBLE("epsilon"),
        /* rangeObjectsZ     = */ CFGDOUBLE("range_object_z"), // in meters
        /* widthMargin       = */ CFGDOUBLE("width_margin"),
        /* maxSections       = */ CFGDOUBLE("max_sections")
    };

    const DisparityParameters disparity_parameters {
        /* maxDisparity         = */ MAX_DISPARITY,
        /* sigmaDisparityObject = */ CFGDOUBLE("sigma_object"),
        /* sigmaDisparityGround = */ CFGDOUBLE("sigma_ground"),
        /* sigmaSky             = */ CFGDOUBLE("sigma_sky") // Should be small compared to sigma_dis
    };

    EstimatedCameraParameters estimated_camera_parameters = {
        /* sigmaCameraTilt   = */ CFGDOUBLE("sigma_tilt") DEGREE,
        /* sigmaCameraHeight = */ CFGDOUBLE("sigma_height")
    };

    const float max_dis_display = CFGDOUBLE("max_dis_disparity");
    const float disparity_sky = CFGDOUBLE("sky_disparity");


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
    road_estimation.SetCameraParameters(camera_parameters, road_estimation_parameters);
    stixles.SetParameters(probabilities_parameters, camera_parameters, disparity_parameters, stixel_model_parameters);

    cv::Mat left_frame_input, right_frame_input;
    cv::Mat left_frame_rect, right_frame_rect;
    cv::Mat left_frame_color, right_frame_color, left_frame_grey, right_frame_grey;
    cv::Mat disparity_im, disparity_im_color;
    cv::Size frame_size = cv::Size(0, 0);
    cv::Size rect_size = cv::Size(0, 0);
    cv::Rect rect_roi_up = cv::Rect(0, 0, 0, 0);
    cv::Rect rect_roi_down = cv::Rect(0, 0, 0, 0);
    cv::Rect disp_roi_up = cv::Rect(0, 0, 0, 0);
    cv::Rect disp_roi_down = cv::Rect(0, 0, 0, 0);
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::Mat Rl, Rr, Pl, Pr, Q;

    cv::Mat mix_frame;
    cv::namedWindow("Test");

    if (video_alignment > 0) {
        for (int i = 0; i < video_alignment; i++)
            right_video.grab();
    }
    else if (video_alignment < 0) {
        for (int i = 0; i < -video_alignment; i++)
            left_video.grab();
    }

    try
    {
        int currentFrame = 0;
        while (currentFrame < 12000)
        {
            if (!left_video.read(left_frame_input) || !right_video.read(right_frame_input))
            {
                std::cerr << "Reach the end of video file" << std::endl;
                break;
            }
            currentFrame++;
            if (left_frame_input.size() != right_frame_input.size()) {
                std::cerr << "Both images must have the same dimensions" << std::endl;
                return EXIT_FAILURE;
            }

            cv::resize(left_frame_input, left_frame_input, cv::Size(0, 0), prescale, prescale);
            cv::resize(right_frame_input, right_frame_input, cv::Size(0, 0), prescale, prescale);

            if (left_frame_input.size() != frame_size)
            {
                frame_size = left_frame_input.size();
                if (need_rect) {
                    stereoRectify(cameraMatrixL, cameraDistCoeffL, cameraMatrixR, cameraDistCoeffR, frame_size, cameraRotationMatrix, cameraTranslationMatrix,
                        Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 0);
                    initUndistortRectifyMap(cameraMatrixL, cameraDistCoeffL, Rl, Pl, frame_size, CV_32FC1, mapLx, mapLy);
                    initUndistortRectifyMap(cameraMatrixR, cameraDistCoeffR, Rr, Pr, frame_size, CV_32FC1, mapRx, mapRy);
                }
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

            if (need_rect) {
                remap(left_frame_input, left_frame_rect, mapLx, mapLy, cv::INTER_LINEAR);
                remap(right_frame_input, right_frame_rect, mapRx, mapRy, cv::INTER_LINEAR);
            }
            else {
                left_frame_rect = left_frame_input;
                right_frame_rect = right_frame_input;
            }

            left_frame_rect = left_frame_rect(rect_roi_up);
            right_frame_rect = right_frame_rect(rect_roi_up);

            if (left_frame_rect.channels() > 1) {
                left_frame_color = left_frame_rect;
                cv::cvtColor(left_frame_rect, left_frame_grey, CV_BGR2GRAY);
            }
            else {
                cv::cvtColor(left_frame_rect, left_frame_color, CV_GRAY2BGR);
                left_frame_grey = left_frame_rect;
            }

            if (right_frame_rect.channels() > 1) {
                right_frame_color = right_frame_rect;
                cv::cvtColor(right_frame_rect, right_frame_grey, CV_BGR2GRAY);
            }
            else {
                cv::cvtColor(right_frame_rect, right_frame_color, CV_GRAY2BGR);
                right_frame_grey = right_frame_rect;
            }
            disparity_estimation.LoadImages(left_frame_grey, right_frame_grey);
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
            int realCols = stixles.FetchRealCols();

            cv::Mat left_frame_stx;
            ShowStixels(left_frame_color, left_frame_stx, stx, stixel_model_parameters, realCols, estimated_camera_parameters.horizonPoint, max_dis_display, disparity_sky);

            //MIX IMGS TOGETHER
            //left_frame_color.copyTo(mix_frame(rect_roi_up));
            //cv::cvtColor(disparity_im, disparity_im_color, CV_GRAY2BGR);
            left_frame_stx.copyTo(mix_frame(rect_roi_up));
            disparity_im_color.copyTo(mix_frame(rect_roi_down));
            cv::imshow("Test", mix_frame);

            char c = cv::waitKey(1);
            if (c == 27) break;
            if (c != -1)
            {
                std::cout << "PAUSED" << c << std::endl;
                cv::waitKey(0);
            }

        }
    }
    catch (const std::exception&)
    {
            
    }

    stixles.Finish();
    road_estimation.Finish();
    disparity_estimation.Finish();
    cv::waitKey();
    return 0;
}
