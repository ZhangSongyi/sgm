#ifndef STRUCT_H_
#define STRUCT_H_

struct Section {
    int type;
    int vB, vT;
    float disparity;
};

struct CameraParameters {
    float cameraCenterX;
    float cameraCenterY;         ///< Image center from stereo camera
    float baseline;              ///< Stereo camera baseline
    float focal;                 ///< Stereo camera focal length
};

struct EstimatedCameraParameters {
    float horizonPoint;          ///< Horizon point of v-disparity histogram
    float pitch;                 ///< Camera pitch
    float cameraHeight;          ///< Camera height
    float slope;
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

#endif /* STRUCT_H_ */