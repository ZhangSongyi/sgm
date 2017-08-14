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

#endif /* STRUCT_H_ */