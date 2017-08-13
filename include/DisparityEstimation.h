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

class DisparityEstimation
{
public:
    DisparityEstimation();
    ~DisparityEstimation();
    void Initialize(const uint8_t p1, const uint8_t p2);
    cv::Mat Compute(cv::Mat left, cv::Mat right, float *elapsed_time_ms);
    void Finish();
private:
    class DisparityEstimationImpl;
    std::auto_ptr<DisparityEstimationImpl> m_impl;
};

#endif /* DISPARITY_METHOD_H_ */
