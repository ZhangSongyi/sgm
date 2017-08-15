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


#include <opencv2/opencv.hpp>
#include <memory>
#include "configuration.h"
#include "struct.h"

#ifndef STIXELS_H_
#define STIXELS_H_

#define INVALID_DISPARITY	128.0f
#define MAX_LOGPROB			10000.0f

#define GROUND	0
#define OBJECT	1
#define SKY		2

class Stixels {
public:
	Stixels();
	~Stixels();
	void Initialize();
	float Compute();
	Section* GetStixels();
	int GetRealCols();
	int GetMaxSections();
	void Finish();
    void LoadDisparityImage(const pixel_t* m_disp, cv::Size imageSize, EstimatedCameraParameters estimated_camera_params);
    void LoadDisparityImageD(pixel_t* d_disp, cv::Size imageSize, EstimatedCameraParameters estimated_camera_params);
	void SetDisparityImage(pixel_t *disp_im);
	void SetProbabilities(ProbabilitiesParameters params);
	void SetCameraParameters(CameraParameters camera_params, EstimatedCameraParameters estimated_camera_params);
    void SetDisparityParameters(const int rows, const int cols, const int max_dis,
    		const float sigma_disparity_object, const float sigma_disparity_ground, float sigma_sky);
	void SetModelParameters(StixelModelParameters stixel_model_parameters);
private:
    class StixelsImpl;
    std::auto_ptr<StixelsImpl> m_impl;
};

#endif /* STIXELS_H_ */
