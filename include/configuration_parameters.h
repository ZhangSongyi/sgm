#ifndef CONFIGURATION_PARAMETERS_H_
#define CONFIGURATION_PARAMETERS_H_

#include <stdint.h>
#include "struct.h"
/**
 ** DISPARITY PARAMETERS CONFIGURATION
 **/
 const int p1 = 32;
 const int p2 = 128;

/**
 ** ROAD ESTIMATION CONFIGURATION
 **/

const CameraParameters camera_parameters = {
    /*cameraCenterX      = */ 651.216186523f,
    /*cameraCenterY      = */ 224.2f,
    /*baseline           = */ 0.643f,
    /*focal              = */ 960.0f
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