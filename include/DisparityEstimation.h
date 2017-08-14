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

#ifndef DISPARITY_ESTIMATION_H_
#define DISPARITY_ESTIMATION_H_

#include <opencv2/opencv.hpp>
#include <memory>
#include "configuration.h"

class DisparityEstimation
{
public:
    DisparityEstimation();
    ~DisparityEstimation();
    void Initialize();
    void SetParameter(const uint8_t p1, const uint8_t p2);
    void SetColorTable(const uint8_t color_table[MAX_DISPARITY * 3]);
    void LoadImages(cv::Mat left, cv::Mat right);
    void LoadImagesD(uint8_t* left, uint8_t* right, cv::Size imageSize);
    void Compute(float *elapsed_time_ms);
    cv::Mat FetchDisparityResult();
    cv::Mat FetchColoredDisparityResult();
    uint8_t* FetchDisparityResultD();
    pixel_t* FetchDisparityResultPixelD();
    void Finish();
private:
    class DisparityEstimationImpl;
    std::auto_ptr<DisparityEstimationImpl> m_impl;
};

#endif /* DISPARITY_METHOD_H_ */
