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
#include "MedianFilterKernels.cuh"
#include "CostAggregationKernels.hpp"
#include "debug.h"

class DisparityEstimationImpl
{
public:
    DisparityEstimationImpl() {}
    ~DisparityEstimationImpl() {}
    void Initialize(const uint8_t p1, const uint8_t p2);
    cv::Mat Compute(cv::Mat left, cv::Mat right, float *elapsed_time_ms);
    void Finish();

private: /*CUDA Host Pointer*/
    uint8_t *h_disparity;
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

private:
    void malloc_memory();
    void free_memory();
};

void DisparityEstimationImpl::Initialize(const uint8_t p1, const uint8_t p2) {
    // We are not using shared memory, use L1
    //CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    //CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Create streams
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
    first_alloc = true;
    this->p1 = p1;
    this->p2 = p2;
    rows = 0;
    cols = 0;
}

cv::Mat DisparityEstimationImpl::Compute(cv::Mat left, cv::Mat right, float *elapsed_time_ms) {
	if(cols != left.cols || rows != left.rows) {
		debug_log("WARNING: cols or rows are different");
		if(!first_alloc) {
			debug_log("Freeing memory");
			free_memory();
		}
		first_alloc = false;
		cols = left.cols;
		rows = left.rows;
		size = rows*cols;
		size_cube_l = size*MAX_DISPARITY;
        malloc_memory();
	}
	debug_log("Copying images to the GPU");
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));

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

	debug_log("Copying final disparity to CPU");
	CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));

	cv::Mat disparity(rows, cols, CV_8UC1, h_disparity);
	return disparity;
}

void DisparityEstimationImpl::Finish() {
	if(!first_alloc) {
		free_memory();
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream2));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream3));
	}
}

void DisparityEstimationImpl::malloc_memory() {
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
    h_disparity = new uint8_t[size];
}

void DisparityEstimationImpl::free_memory() {
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
	CUDA_CHECK_RETURN(cudaFree(d_cost));

	delete[] h_disparity;
}

DisparityEstimation::DisparityEstimation() : 
    m_impl(new DisparityEstimationImpl()) {}

DisparityEstimation::~DisparityEstimation() {}

void DisparityEstimation::Initialize(const uint8_t p1, const uint8_t p2) {
    m_impl->Initialize(p1, p2);
}

cv::Mat DisparityEstimation::Compute(cv::Mat left, cv::Mat right, float *elapsed_time_ms) {
    return m_impl->Compute(left, right, elapsed_time_ms);
}

void DisparityEstimation::Finish() {
    m_impl->Finish();
}
