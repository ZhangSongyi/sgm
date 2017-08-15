#ifndef STRUCT_H_
#define STRUCT_H_

struct Section {
    int type;
    int vB, vT;
    float disparity;
};

struct CameraParameters {
    float cameraCenterX;
    float cameraCenterY;   ///< Image center from stereo camera
    float baseline;        ///< Stereo camera baseline
    float focal;           ///< Stereo camera focal length
};

struct EstimatedCameraParameters {
    float sigmaCameraTilt;
    float sigmaCameraHeight;
    float horizonPoint;          ///< Horizon point of v-disparity histogram
    float pitch;                 ///< Camera pitch
    float cameraHeight;          ///< Camera height
    float slope;
};

struct ProbabilitiesParameters {
    float out;
    float outSky;
    float groundGivenNExist;
    float objectGivenNExist;
    float skyGivenNExist;
    float nExistDis;
    float ground;
    float object;
    float sky;
    float ord;
    float grav;
    float blg;
};

struct ExportProbabilitiesParameters {
    float uniformSky;
    float uniform;
    float nExistsGivenSkyLOG;
    float nExistsGivenSkyNLOG;
    float nExistsGivenGroundLOG;
    float nExistsGivenGroundNLOG;
    float nExistsGivenObjectLOG;
    float nExistsGivenObjectNLOG;
};

struct StixelModelParameters {
    int columnStep;
    int medianStep;
    float epsilon;
    float rangeObjectsZ;
    int widthMargin;
};

struct DisparityParameters {
    int maxDisparity;
    float sigmaDisparityObject;
    float sigmaDisparityGround;
    float sigmaSky;
};

struct StixelParameters {
    int rows;
    int rows_power2;
    int cols;
    int max_dis;
    float rows_log;
    float normalization_sky;
    float inv_sigma2_sky;
    float max_dis_log;
    int max_sections;
    StixelModelParameters modelParameters;
    CameraParameters cameraParameters;
    EstimatedCameraParameters estimatedCameraParameters;
    ProbabilitiesParameters probabilitiesParameters;
    ExportProbabilitiesParameters exportProbabilitiesParameters;
};

#endif /* STRUCT_H_ */