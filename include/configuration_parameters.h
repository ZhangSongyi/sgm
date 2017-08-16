#ifndef CONFIGURATION_PARAMETERS_H_
#define CONFIGURATION_PARAMETERS_H_

#include <stdint.h>
#include "struct.h"
#include <opencv2/core.hpp>

/*
** STEREO CONFIGURATION
*/

const float prescale = 0.5;

/*
** [ fx,  0, cx ]
** [  0, fy, cy ]
** [  0,  0,  1 ]
*/
const cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) <<
    1913.570357870300, 0000.000000000000, 978.2503247295590 * prescale,
    0000.000000000000, 1911.673741716337, 597.5234837578698 * prescale,
    0000.000000000000, 0000.000000000000, 001.0000000000000);
const cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) <<
    1919.794575060740, 0000.000000000000, 978.0987810362845 * prescale,
    0000.000000000000, 1917.341981391412, 598.3637896639764 * prescale,
    0000.000000000000, 0000.000000000000, 001.0000000000000);

/*
** [ k1, k2 ]
** [ p1, p2 ]
** [ k3     ]
*/
const cv::Mat cameraDistCoeffL = (cv::Mat_<double>(1, 5) <<
    -.446625254806059, +.263269797805976,
    +.001203668229826, -.000834763926887,
    -.071769704566470);
const cv::Mat cameraDistCoeffR = (cv::Mat_<double>(1, 5) <<
    -.447441627431729, +.270363885260030,
    +.000812872676857, -.001077810656806,
    -.068076999198103);

const cv::Mat cameraRotationMatrix = (cv::Mat_<double>(3, 3) <<
    +.999792356761562, +.014694749710781, +.014114117637624,
    -.014705601526619, +.999891645550221, +.000665163597689,
    -.014106333503107, -.000872633834369, +.999900119944632);

const cv::Mat cameraTranslationMatrix = (cv::Mat_<double>(3, 1) <<
    +412.9784450489922,
    -005.8885085059490,
    +000.0337644648471);

/*
** DISPARITY PARAMETERS CONFIGURATION
*/
 const int p1 = 32;
 const int p2 = 128;

/*
** ROAD ESTIMATION CONFIGURATION
*/

//const CameraParameters camera_parameters = {
//     /*cameraCenterX      = */ 651.216186523f,
//     /*cameraCenterY      = */ 224.2f,
//     /*baseline           = */ 0.643f,
//     /*focal              = */ 960.0f
//};

const CameraParameters camera_parameters = {
    /*cameraCenterX      = */ cameraMatrixL.at<double>(0, 2),
    /*cameraCenterY      = */ cameraMatrixL.at<double>(1, 2),
    /*baseline           = */ cameraTranslationMatrix.at<double>(0, 0) / 1000,
    /*focal              = */ cameraMatrixL.at<double>(0, 0)
};

const ProbabilitiesParameters probabilities_parameters = {
    /*out                = */ 0.15f,
    /*outSky             = */ 0.4f,
    /*groundGivenNExist  = */ 0.36f,
    /*objectGivenNExist  = */ 0.28f,
    /*skyGivenNExist     = */ 0.36f,
    /*nExistDis          = */ 0.0f,
    /*ground             = */ 1.0f / 3.0f,
    /*object             = */ 1.0f / 3.0f,
    /*sky                = */ 1.0f / 3.0f,
    /*ord                = */ 0.2f,
    /*grav               = */ 0.1f,
    /*blg                = */ 0.04f
};

const StixelModelParameters stixel_model_parameters = {
    /* columnStep        = */ 5,
    /* medianStep        = */ false,
    /* epsilon           = */ 3.0f,
    /* rangeObjectsZ     = */ 10.20f, // in meters
    /* widthMargin       = */ 0,
    /* maxSections       = */ 50
};

struct DisparityParameters disparity_parameters {
    /* maxDisparity         = */ 128,
    /* sigmaDisparityObject = */ 1.0f, 
    /* sigmaDisparityGround = */ 2.0f,
    /* sigmaSky             = */ 0.1f // Should be small compared to sigma_dis
};

EstimatedCameraParameters estimated_camera_parameters = {
    /*sigmaCameraTilt    = */ 0.05f * (PIFLOAT) / 180.0f,
    /*sigmaCameraHeight  = */ 0.05f
};

const float max_dis_display = (float)30;



#endif /* CONFIGURATION_PARAMETERS_H_ */