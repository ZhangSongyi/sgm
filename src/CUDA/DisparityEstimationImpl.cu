/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#include "DisparityEstimation.h"
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "util.hpp"
#include "configuration.h"
#include "CenterSymmetricCensusKernel.cuh"
#include "HammingDistanceCostKernel.cuh"
#include "ImageProcessorKernels.cuh"
#include "CostAggregationKernels.hpp"
#include "debug.h"

class DisparityEstimation::DisparityEstimationImpl
{
public:
    DisparityEstimationImpl() {}
    ~DisparityEstimationImpl() {}
    void Initialize();
    void SetParameter(const uint8_t p1, const uint8_t p2);
    void SetColorTable(const uint8_t color_table[MAX_DISPARITY * 3]);
    void LoadImages(cv::Mat left, cv::Mat right);
    void LoadImagesD(uint8_t* left, uint8_t* right, cv::Size imageSize);
    void Compute(float *elapsed_time_ms);
    cv::Mat FetchDisparityResult();
    cv::Mat FetchColoredDisparityResult();

    uint8_t * FetchDisparityResultD() { return d_disparity_filtered_uchar; }
    pixel_t * FetchDisparityResultPixelD();
    void Finish();

private:
    void LoadImagesCore(const uint8_t* d_left, const uint8_t* d_right, cv::Size imageSize, cudaMemcpyKind dir);

private: /*CUDA Host Pointer*/
    uint8_t *h_disparity;
    uint8_t *h_disparity_colored;
    uint8_t p1, p2;
    bool first_alloc;
    uint32_t cols, rows, size, size_cube_l;

private: /*CUDA Device Pointer*/
    cudaStream_t stream1, stream2, stream3;//, stream4, stream5, stream6, stream7, stream8;
    uint8_t *d_im0;
    uint8_t *d_im1;
    cost_t *d_transform0;
    cost_t *d_transform1;
    uint8_t *d_cost;
    uint8_t *d_disparity;
    uint8_t *d_disparity_filtered_uchar;
    uint8_t *d_disparity_filtered_colored_uchar;
    pixel_t *d_disparity_filtered_pixel;
    uint16_t *d_S;
    uint8_t *d_L0;
    uint8_t *d_L1;
    uint8_t *d_L2;
    uint8_t *d_L3;
    uint8_t *d_L4;
    uint8_t *d_L5;
    uint8_t *d_L6;
#if PATH_AGGREGATION == 8
    uint8_t *d_L7;
#endif
    uint8_t *d_color_table;

private:
    void malloc_memory();
    void free_memory();
};

void DisparityEstimation::DisparityEstimationImpl::Initialize() {
    // We are not using shared memory, use L1
    //CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    //CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    //Create default color table
    uint8_t h_color_table[MAX_DISPARITY * 3];
    float color_table_section = MAX_DISPARITY / 8.0;
    float scale_factor = 1024.0 / MAX_DISPARITY;
    for (int i = 0; i < MAX_DISPARITY; i++)
    {
        h_color_table[i * 3 + 0] = std::max(std::min((int)(std::min(i - color_table_section * (-1), color_table_section * 5 - i) * scale_factor), 255), 0);//b
        h_color_table[i * 3 + 1] = std::max(std::min((int)(std::min(i - color_table_section * 1,    color_table_section * 7 - i) * scale_factor), 255), 0);//g
        h_color_table[i * 3 + 2] = std::max(std::min((int)(std::min(i - color_table_section * 3,    color_table_section * 9 - i) * scale_factor), 255), 0);//r
    }
    // Create streams
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_color_table, sizeof(uint8_t)*MAX_DISPARITY * 3));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_color_table, h_color_table, MAX_DISPARITY * 3, cudaMemcpyHostToDevice, stream1));
    first_alloc = true;
    rows = 0;
    cols = 0;
}

void DisparityEstimation::DisparityEstimationImpl::SetParameter(const uint8_t _p1, const uint8_t _p2)
{
    if (_p1 < _p2) {
        p1 = _p1;
        p2 = _p2;
    }
}

void DisparityEstimation::DisparityEstimationImpl::SetColorTable(const uint8_t color_table[MAX_DISPARITY * 3])
{
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, color_table, MAX_DISPARITY * 3, cudaMemcpyHostToDevice, stream1));
}

inline void DisparityEstimation::DisparityEstimationImpl::LoadImagesCore(const uint8_t* left, const uint8_t* right, cv::Size imageSize, cudaMemcpyKind dir)
{
    if (cols != imageSize.width || rows != imageSize.height) {
        debug_log("WARNING: cols or rows are different");
        if (!first_alloc) {
            debug_log("Freeing memory");
            free_memory();
        }
        first_alloc = false;
        cols = imageSize.width;
        rows = imageSize.height;
        size = rows*cols;
        size_cube_l = size*MAX_DISPARITY;
        malloc_memory();
    }
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left, sizeof(uint8_t)*size, dir, stream1));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right, sizeof(uint8_t)*size, dir, stream1));
}

void DisparityEstimation::DisparityEstimationImpl::LoadImages(cv::Mat left, cv::Mat right) {
    LoadImagesCore(left.ptr<uint8_t>(), right.ptr<uint8_t>(), left.size(), cudaMemcpyHostToDevice);
}

void DisparityEstimation::DisparityEstimationImpl::LoadImagesD(uint8_t* d_left, uint8_t* d_right, cv::Size imageSize) {
    LoadImagesCore(d_left, d_right, imageSize, cudaMemcpyDeviceToDevice);
}

void DisparityEstimation::DisparityEstimationImpl::Compute(float *elapsed_time_ms) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (cols+block_size.x-1) / block_size.x;
	grid_size.y = (rows+block_size.y-1) / block_size.y;

	debug_log("Calling CSCT");
	CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);

	// Hamming distance
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	debug_log("Calling Hamming Distance");
	HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);

	// Cost Aggregation
	const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

	debug_log("Calling Left to Right");
	CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	debug_log("Calling Right to Left");
	CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	debug_log("Calling Up to Down");
	CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	debug_log("Calling Down to Up");
	CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

#if PATH_AGGREGATION == 8
	CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

	CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
#endif
	debug_log("Calling Median Filter");
	MedianFilter3x3<<<(size+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);
    
	cudaEventRecord(stop, 0);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	cudaEventElapsedTime(elapsed_time_ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

cv::Mat DisparityEstimation::DisparityEstimationImpl::FetchDisparityResult()
{
    debug_log("Copying final disparity to CPU");
    CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

    cv::Mat disparity(rows, cols, CV_8UC1, h_disparity);
    return disparity;
}

cv::Mat DisparityEstimation::DisparityEstimationImpl::FetchColoredDisparityResult()
{
    ColorizeDisparityImage<<<(size + MAX_DISPARITY - 1) / MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity_filtered_uchar, d_disparity_filtered_colored_uchar, d_color_table, rows, cols);

    debug_log("Copying final disparity to CPU");
    CUDA_CHECK_RETURN(cudaMemcpy(h_disparity_colored, d_disparity_filtered_colored_uchar, sizeof(uint8_t)*size*3, cudaMemcpyDeviceToHost));

    cv::Mat disparity_colored(rows, cols, CV_8UC3, h_disparity_colored);
    return disparity_colored;
}

pixel_t* DisparityEstimation::DisparityEstimationImpl::FetchDisparityResultPixelD() {
    TypeConvert<<<(size + MAX_DISPARITY - 1) / MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity_filtered_uchar, d_disparity_filtered_pixel, rows, cols);
    cudaDeviceSynchronize();
    return d_disparity_filtered_pixel;
}

void DisparityEstimation::DisparityEstimationImpl::Finish() {
	if(!first_alloc) {
		free_memory();
        CUDA_CHECK_RETURN(cudaFree(d_color_table));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream2));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream3));
	}
}

void DisparityEstimation::DisparityEstimationImpl::malloc_memory() {
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im0, sizeof(uint8_t)*size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im1, sizeof(uint8_t)*size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L0, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L1, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L2, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L3, sizeof(uint8_t)*size_cube_l));
#if PATH_AGGREGATION == 8
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L4, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L5, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L6, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L7, sizeof(uint8_t)*size_cube_l));
#endif
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, sizeof(uint8_t)*size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_pixel, sizeof(pixel_t)*size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_colored_uchar, sizeof(uint8_t)*size*3));
    h_disparity = new uint8_t[size];
    h_disparity_colored = new uint8_t[size*3];
}

void DisparityEstimation::DisparityEstimationImpl::free_memory() {
	CUDA_CHECK_RETURN(cudaFree(d_im0));
	CUDA_CHECK_RETURN(cudaFree(d_im1));
	CUDA_CHECK_RETURN(cudaFree(d_transform0));
	CUDA_CHECK_RETURN(cudaFree(d_transform1));
	CUDA_CHECK_RETURN(cudaFree(d_L0));
	CUDA_CHECK_RETURN(cudaFree(d_L1));
	CUDA_CHECK_RETURN(cudaFree(d_L2));
	CUDA_CHECK_RETURN(cudaFree(d_L3));
#if PATH_AGGREGATION == 8
	CUDA_CHECK_RETURN(cudaFree(d_L4));
	CUDA_CHECK_RETURN(cudaFree(d_L5));
	CUDA_CHECK_RETURN(cudaFree(d_L6));
	CUDA_CHECK_RETURN(cudaFree(d_L7));
#endif
	CUDA_CHECK_RETURN(cudaFree(d_disparity));
	CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_uchar));
    CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_pixel));
    CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_colored_uchar));
	CUDA_CHECK_RETURN(cudaFree(d_cost));

	delete[] h_disparity;
    delete[] h_disparity_colored;
}

DisparityEstimation::DisparityEstimation() : 
    m_impl(new DisparityEstimationImpl()) {}

DisparityEstimation::~DisparityEstimation() {}

void DisparityEstimation::Initialize() {
    m_impl->Initialize();
}

void DisparityEstimation::SetParameter(const uint8_t p1, const uint8_t p2) {
    m_impl->SetParameter(p1, p2);
}

void DisparityEstimation::SetColorTable(const uint8_t color_table[MAX_DISPARITY * 3]) {
    m_impl->SetColorTable(color_table);
}

void DisparityEstimation::LoadImages(cv::Mat left, cv::Mat right) {
    m_impl->LoadImages(left, right);
}

void DisparityEstimation::LoadImagesD(uint8_t* left, uint8_t* right, cv::Size imageSize) {
    m_impl->LoadImagesD(left, right, imageSize);
}

void DisparityEstimation::Compute(float *elapsed_time_ms) {
    m_impl->Compute(elapsed_time_ms);
}

cv::Mat DisparityEstimation::FetchDisparityResult() {
    return m_impl->FetchDisparityResult();
}

cv::Mat DisparityEstimation::FetchColoredDisparityResult() {
    return m_impl->FetchColoredDisparityResult();
}

uint8_t* DisparityEstimation::FetchDisparityResultD() {
    return m_impl->FetchDisparityResultD();
}

float* DisparityEstimation::FetchDisparityResultPixelD() {
    return m_impl->FetchDisparityResultPixelD();
}

void DisparityEstimation::Finish() {
    m_impl->Finish();
}
