#ifndef CONFIGURATION_PARAMETERS_H_
#define CONFIGURATION_PARAMETERS_H_

#include <stdint.h>
/**
 ** DISPARITY PARAMETERS CONFIGURATION
 **/
 const int p1 = 32;
 const int p2 = 128;

/**
 ** ROAD ESTIMATION CONFIGURATION
 **/

/* Disparity Parameters */
const float sigma_disparity_object = 1.0f;
const float sigma_disparity_ground = 2.0f;
const float sigma_sky = 0.1f; // Should be small compared to sigma_dis

/* Probabilities */
const float pout = 0.15f;
const float pout_sky = 0.4f;
const float pord = 0.2f;
const float pgrav = 0.1f;
const float pblg = 0.04f;

//
// Must add 1
const float pground_given_nexist = 0.36f;
const float pobject_given_nexist = 0.28f;
const float psky_given_nexist = 0.36f;

const float pnexist_dis = 0.0f;
const float pground = 1.0f / 3.0f;
const float pobject = 1.0f / 3.0f;
const float psky = 1.0f / 3.0f;

// Virtual parameters
const float focal = 960.0f;
const float baseline = 0.643f;
const float camera_center_y = 224.2f;
const int column_step = 5;
const int width_margin = 0;

const float sigma_camera_tilt = 0.05f;
const float sigma_camera_height = 0.05f;
const float camera_center_x = 651.216186523f;

/* Model Parameters */
const bool median_step = false;
const float epsilon = 3.0f;
const float range_objects_z = 10.20f; // in meters

const float max_dis_display = (float)30;

#endif /* CONFIGURATION_PARAMETERS_H_ */