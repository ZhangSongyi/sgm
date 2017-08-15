/**
    This file is part of stixels. (https://github.com/dhernandez0/stixels).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    stixels is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    stixels is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with stixels.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef ROAD_ESTIMATION_H_
#define ROAD_ESTIMATION_H_

#include <opencv2/opencv.hpp>
#include <memory>
#include "configuration.h"
#include "struct.h"

class RoadEstimation
{
public:
	RoadEstimation();
	~RoadEstimation();
	void Initialize();
    void SetCameraParameters(const CameraParameters param);
    void LoadDisparityImage(const pixel_t* m_disp, cv::Size imageSize);
    void LoadDisparityImageD(pixel_t* d_disp, cv::Size imageSize);
	bool Compute();
	void Finish();
    EstimatedCameraParameters FetchEstimatedCameraParameters();
private:
    class RoadEstimationImpl;
    std::auto_ptr<RoadEstimationImpl> m_impl;
};
#endif // ROAD_ESTIMATION_H_
