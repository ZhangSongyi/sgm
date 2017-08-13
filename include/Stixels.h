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

#ifndef STIXELS_H_
#define STIXELS_H_

#define PIFLOAT 3.1416f

#define INVALID_DISPARITY	128.0f
#define MAX_LOGPROB			10000.0f

#define GROUND	0
#define OBJECT	1
#define SKY		2

struct Section {
	int type;
	int vB, vT;
	float disparity;
};

struct StixelParameters {
	int vhor;
	int rows;
	int rows_power2;
	int cols;
	int max_dis;
	float rows_log;
	float pnexists_given_sky_log;
	float normalization_sky;
	float inv_sigma2_sky;
	float puniform_sky;
	float nopnexists_given_sky_log;
	float pnexists_given_ground_log;
	float puniform;
	float nopnexists_given_ground_log;
	float pnexists_given_object_log;
	float nopnexists_given_object_log;
	float baseline;
	float focal;
	float range_objects_z;
	float pord;
	float epsilon;
	float pgrav;
	float pblg;
	float max_dis_log;
	int max_sections;
	int width_margin;
};

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
	void SetDisparityImage(pixel_t *disp_im);
	void SetProbabilities(float pout, float pout_sky, float pground_given_nexist,
			float pobject_given_nexist, float psky_given_nexist, float pnexist_dis, float pground,
			float pobject, float psky, float pord, float pgrav, float pblg);
	void SetCameraParameters(int vhor, float focal, float baseline, float camera_tilt,
			float sigma_camera_tilt, float camera_height, float sigma_camera_height, float alpha_ground);
    void SetDisparityParameters(const int rows, const int cols, const int max_dis,
    		const float sigma_disparity_object, const float sigma_disparity_ground, float sigma_sky);
	void SetModelParameters(const int column_step, const bool median_step, float epsilon, float range_objects_z,
			int width_margin);
private:
    class StixelsImpl;
    std::auto_ptr<StixelsImpl> m_impl;
};

#endif /* STIXELS_H_ */
