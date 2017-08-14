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

#include "RoadEstimation.h"
#include "RoadEstimationKernels.cuh"
#include "util.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "debug.h"

class RoadEstimation::RoadEstimationImpl
{
public:
    RoadEstimationImpl();
    ~RoadEstimationImpl();
    void Initialize();
    void SetCameraParameters(const CameraParameters param);
    void LoadImages(const pixel_t* m_disp, cv::Size imageSize);
    void LoadImagesD(pixel_t* d_disp, cv::Size imageSize);
    bool Compute();
    void Finish();
    EstimatedCameraParameters FetchEstimatedCameraParameters() { return m_estimatedCameraParameters; }
private:
    void ComputeCameraProperties(cv::Mat vDisp, const float rho, const float theta, EstimatedCameraParameters& roadParams) const;
    bool ComputeHough(uint8_t *vDisp, float& rho, float& theta, EstimatedCameraParameters& roadParams);

private: /*CUDA Host Pointer*/
    CameraParameters m_cameraParameters;
    EstimatedCameraParameters m_estimatedCameraParameters;

    int m_rangeAngleX;          ///< Angle interval to discard horizontal planes
    int m_rangeAngleY;          ///< Angle interval to discard vertical planes
    int m_HoughAccumThr;        ///< Threshold of the min number of points to form a line
    float m_binThr;             ///< Threshold to binarize vDisparity histogram
    float m_maxPitch;			///< Angle elevation maximun of camera
    float m_minPitch;			///< Angle elevation minimum of camera
    float m_maxCameraHeight;    ///< Height maximun of camera
    float m_minCameraHeight;    ///< Height minimun of camera
    int m_rows;
    int m_cols;

    // Member objects
    float m_rho;                 ///< Line in polar (Distance from (0,0) to the line)
    float m_theta;               ///< Line in polar (Angle of the line with x axis)

    bool first_alloc = false;

private: /*CUDA Device Pointer*/
    pixel_t *d_disparity;
    pixel_t *d_disparity_input;
    int *d_vDisp;
    int *d_maximum;
    uint8_t *d_vDispBinary;
    uint8_t *m_vDisp;

    void malloc_memory();
    void free_memory();
};

RoadEstimation::RoadEstimationImpl::RoadEstimationImpl() {}

RoadEstimation::RoadEstimationImpl::~RoadEstimationImpl() {}

void RoadEstimation::RoadEstimationImpl::Initialize() {
    // Default configuration
    m_rangeAngleX = 5;
    m_rangeAngleY = 5;
    m_HoughAccumThr = 25;
    m_binThr = 0.5f;
    m_maxPitch = 50;
    m_minPitch = -50;
    m_maxCameraHeight = 1.90f;
    m_minCameraHeight = 1.30f;
    m_maxPitch = m_maxPitch*(float)CV_PI / 180.0f;
    m_minPitch = m_minPitch*(float)CV_PI / 180.0f;

    first_alloc = true;
}

void RoadEstimation::RoadEstimationImpl::SetCameraParameters(const CameraParameters param) {
    m_cameraParameters = param;
}

void RoadEstimation::RoadEstimationImpl::Finish() {
    free_memory();
}

void RoadEstimation::RoadEstimationImpl::LoadImages(const pixel_t* m_disp, cv::Size imageSize) {
    if (m_cols != imageSize.width || m_rows != imageSize.height) {
        debug_log("WARNING: cols or rows are different");
        if (!first_alloc) {
            debug_log("Freeing memory");
            free_memory();
        }
        first_alloc = false;
        m_cols = imageSize.width;
        m_rows = imageSize.height;
        malloc_memory();
    }
    CUDA_CHECK_RETURN(cudaMemset(d_maximum, 0, 1 * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemset(d_vDisp, 0, MAX_DISPARITY*m_rows * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_disparity, m_disp, m_rows*m_cols * sizeof(pixel_t), cudaMemcpyHostToDevice));
    d_disparity_input = d_disparity;
}

void RoadEstimation::RoadEstimationImpl::LoadImagesD(pixel_t* d_disp, cv::Size imageSize) {
    if (m_cols != imageSize.width || m_rows != imageSize.height) {
        debug_log("WARNING: cols or rows are different");
        if (!first_alloc) {
            debug_log("Freeing memory");
            free_memory();
        }
        first_alloc = false;
        m_cols = imageSize.width;
        m_rows = imageSize.height;
        malloc_memory();
    }
    d_disparity_input = d_disp;
    CUDA_CHECK_RETURN(cudaMemset(d_maximum, 0, 1 * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemset(d_vDisp, 0, MAX_DISPARITY*m_rows * sizeof(int)));
}

bool RoadEstimation::RoadEstimationImpl::Compute() {
	bool ok = false;

	// Compute the vDisparity histogram
	ComputeHistogram<<<(m_rows*m_cols+256-1)/256, 256>>>(d_disparity_input, d_vDisp, m_rows, m_cols, MAX_DISPARITY);
	ComputeMaximum<<<(m_rows*MAX_DISPARITY +256-1)/256, 256>>>(d_vDisp, d_maximum, m_rows, MAX_DISPARITY);
	ComputeBinaryImage<<<(m_rows*MAX_DISPARITY +256-1)/256, 256>>>(d_vDisp, d_vDispBinary, d_maximum, m_binThr,
			m_rows, MAX_DISPARITY);

    // Compute the Hough transform
	float rho, theta, horizonPoint, pitch, cameraHeight, slope;
    EstimatedCameraParameters roadParams;
	if (ComputeHough(d_vDispBinary, rho, theta, roadParams)) {
		m_rho = rho;
		m_theta = theta;
        m_estimatedCameraParameters = roadParams;
		ok = true;
	}
	return ok;
}

bool RoadEstimation::RoadEstimationImpl::ComputeHough(uint8_t *d_vDispBinary, float& rho, float& theta,
    EstimatedCameraParameters& estimatedCameraParams) {
	// Compute the Hough transform
	std::vector<cv::Vec2f> lines;
	cudaMemcpy(m_vDisp, d_vDispBinary, MAX_DISPARITY*m_rows*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cv::Mat vDisp(m_rows, MAX_DISPARITY, CV_8UC1, m_vDisp);
	cv::HoughLines(vDisp, lines, 1.0, CV_PI/180, m_HoughAccumThr);

	// Get the best line from hough
	for (size_t i=0; i<lines.size(); i++) {
		// Get rho and theta
		rho = abs(lines[i][0]);
		theta = lines[i][1];

		// Compute camera position
		ComputeCameraProperties(vDisp, rho, theta, estimatedCameraParams);

		//printf("%f (%f %f) %f (%f %f)\n", pitch, m_minPitch, m_maxPitch, cameraHeight, m_minCameraHeight, m_maxCameraHeight);
		//if (pitch>=m_minPitch && pitch<=m_maxPitch && cameraHeight>=m_minCameraHeight && cameraHeight<=m_maxCameraHeight) {
		if (estimatedCameraParams.pitch>=m_minPitch && estimatedCameraParams.pitch<=m_maxPitch && 
            estimatedCameraParams.cameraHeight >= m_minCameraHeight && estimatedCameraParams.cameraHeight <= m_maxCameraHeight) {
			return true;
		}
	}

	return false;
}

void RoadEstimation::RoadEstimationImpl::ComputeCameraProperties(cv::Mat vDisp, const float rho, const float theta,
    EstimatedCameraParameters& estimatedCameraParams) const
{
	// Compute Horizon Line (2D)
    estimatedCameraParams.horizonPoint = rho/sinf(theta);

	// Compute pitch -> arctan((cy - y0Hough)/focal) It is negative because y axis is inverted
    estimatedCameraParams.pitch = -atanf((m_cameraParameters.cameraCenterY - estimatedCameraParams.horizonPoint)/(m_cameraParameters.focal));

	// Compute the slope needed to compute the Camera height
	float last_row = (float)(vDisp.rows-1);
	float vDispDown = (rho-last_row*sinf(theta))/cosf(theta);
    estimatedCameraParams.slope = (0 - vDispDown)/(estimatedCameraParams.horizonPoint - last_row);

	// Compute the camera height -> baseline*cos(pitch)/slopeHough
    estimatedCameraParams.cameraHeight = m_cameraParameters.baseline*cosf(estimatedCameraParams.pitch)/ estimatedCameraParams.slope;
}

void RoadEstimation::RoadEstimationImpl::malloc_memory() {
    m_vDisp = (uint8_t*)malloc(MAX_DISPARITY*m_rows * sizeof(uint8_t));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, m_cols*m_rows * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_vDisp, MAX_DISPARITY*m_rows * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_maximum, 1 * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_vDispBinary, MAX_DISPARITY*m_rows * sizeof(uint8_t)));
}

void RoadEstimation::RoadEstimationImpl::free_memory() {
    CUDA_CHECK_RETURN(cudaFree(d_vDisp));
    CUDA_CHECK_RETURN(cudaFree(d_disparity));
    CUDA_CHECK_RETURN(cudaFree(d_maximum));
    CUDA_CHECK_RETURN(cudaFree(d_vDispBinary));
    free(m_vDisp);
}

RoadEstimation::RoadEstimation() : 
    m_impl(new RoadEstimationImpl()) {}

RoadEstimation::~RoadEstimation() {}

void RoadEstimation::Initialize() {
    m_impl->Initialize();
}

void RoadEstimation::SetCameraParameters(const CameraParameters param)
{
    m_impl->SetCameraParameters(param);
}

void RoadEstimation::LoadImages(const pixel_t* m_disp, cv::Size imageSize)
{
    m_impl->LoadImages(m_disp, imageSize);
}

void RoadEstimation::LoadImagesD(pixel_t* d_disp, cv::Size imageSize)
{
    m_impl->LoadImagesD(d_disp, imageSize);
}

bool RoadEstimation::Compute() {
    return m_impl->Compute();
}

void RoadEstimation::Finish() {
    m_impl->Finish();
}

EstimatedCameraParameters RoadEstimation::FetchEstimatedCameraParameters() {
    return m_impl->FetchEstimatedCameraParameters();
}
