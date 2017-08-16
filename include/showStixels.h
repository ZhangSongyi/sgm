#ifndef SHOW_STIXELS_H_
#define SHOW_STIXELS_H_

#include <opencv2/core.hpp>
#include "struct.h"

void ShowStixels(cv::Mat left_frame, cv::Mat& left_frame_stx, Section* stx, StixelModelParameters model_parameters, int real_cols, int horizon_point, int max_dis_display);

#endif