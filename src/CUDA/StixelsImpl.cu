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

#include "Stixels.h"
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <nvToolsExt.h>
#include "StixelsKernels.cuh"
#include "util.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Stixels::StixelsImpl {
public:
    StixelsImpl();
    ~StixelsImpl();
    void Initialize();
    void LoadDisparityImage(const pixel_t* m_disp, cv::Size imageSize, EstimatedCameraParameters estimated_camera_params);
    void LoadDisparityImageD(pixel_t* d_disp, cv::Size imageSize, EstimatedCameraParameters estimated_camera_params);
    float Compute();
    Section* GetStixels() { return m_stixels; }
    int GetRealCols() { return m_realcols; }
    int GetMaxSections() { return m_max_sections; }
    void Finish();
    void SetDisparityImage(pixel_t *disp_im);
    void SetProbabilities(ProbabilitiesParameters params);
    void SetCameraParameters(CameraParameters camera_params, EstimatedCameraParameters estimated_camera_params);
    void SetDisparityParameters(const int rows, const int cols, const int max_dis,
        const float sigma_disparity_object, const float sigma_disparity_ground, float sigma_sky);
    void SetModelParameters(StixelModelParameters stixel_model_parameters);

private: /*CUDA Host Pointer*/
    // Variables
    pixel_t *m_disp_im;
    pixel_t *m_disp_im_modified;

    // Probabilities
    ProbabilitiesParameters m_probabilities_params;
    ExportProbabilitiesParameters m_export_probabilities_params;
    CameraParameters m_camera_parameters;
    EstimatedCameraParameters m_estimated_camera_params;
    StixelModelParameters m_model_params;

    // Disparity Parameters
    int m_max_dis;
    float m_max_disf;
    int m_rows, m_cols, m_realcols;
    float m_sigma_disparity_object;
    float m_sigma_disparity_ground;
    float m_sigma_sky;

    // Tables
    float *m_cost_table;
    int16_t *m_index_table;

    // LUTs
    pixel_t *m_sum;
    pixel_t *m_valid;
    float *m_log_lut;
    float *m_obj_cost_lut;

    // Current column data
    pixel_t *m_column;

    // Values of ground function
    float *m_ground_function;

    // Frequently used values
    float m_max_dis_log;
    float m_rows_log;

    // Data Term precomputation
    float m_normalization_sky;
    float m_inv_sigma2_sky;
    float *m_normalization_ground;
    float *m_inv_sigma2_ground;
    float *m_normalization_object;
    float *m_inv_sigma2_object;
    float *m_object_disparity_range;

    // Result
    Section *m_stixels;

private: /*CUDA Device Pointer*/
    pixel_t *d_disparity;
    pixel_t *d_disparity_big;
    float *d_ground_function;
    float *d_normalization_ground;
    float *d_inv_sigma2_ground;
    float *d_normalization_object;
    float *d_inv_sigma2_object;
    float *d_object_lut;
    float *d_object_disparity_range;
    float *d_obj_cost_lut;
    cudaStream_t m_stream1, m_stream2;
    StixelParameters m_params;
    int m_max_sections;
    Section *d_stixels;

private:
    void PrecomputeSky();
    void PrecomputeGround();
    void PrecomputeObject();
    float GetDataCostObject(const float fn, const int dis, const float d);
    float ComputeObjectDisparityRange(const float previous_mean);
    pixel_t ComputeMean(const int vB, const int vT, const int u);
    ExportProbabilitiesParameters ComputeProbabilitiesParameters(ProbabilitiesParameters params,
        int max_dist, int rows);
    float GroundFunction(const int v);
    float FastLog(float v);
    template<typename T>
    void PrintTable(T *table);

private:
    void malloc_memory();
    void free_memory();
    void malloc_image_memory();
    void free_image_memory();
};

Stixels::StixelsImpl::StixelsImpl() {}

Stixels::StixelsImpl::~StixelsImpl() {}

void Stixels::StixelsImpl::Initialize() {
	m_disp_im_modified = m_disp_im;
	m_realcols = (m_cols- m_model_params.widthMargin)/ m_model_params.columnStep;

	m_cost_table = new float[3*m_realcols];
	m_index_table = new int16_t[m_rows*3*m_realcols];

	m_max_sections = 50;

	m_stixels = new Section[m_realcols*m_max_sections];
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_stixels, m_realcols*m_max_sections*sizeof(Section)));

	if(m_model_params.columnStep > 1) {
		CUDA_CHECK_RETURN(cudaMallocHost((void**)&m_disp_im_modified, m_rows*m_realcols*sizeof(float)));
	}

	// Mean precomputation
	m_column = new pixel_t[m_rows*m_realcols];
	m_sum = new pixel_t[(m_rows+1)*m_realcols];
	m_valid = new pixel_t[(m_rows+1)*m_realcols];

	m_ground_function = new float[m_rows];

	// Log LUT, range: 0.0f - 1.0f
    m_log_lut = new float[LOG_LUT_SIZE+1];
	for(int i = 0; i < LOG_LUT_SIZE; i++) {
		const float log_res = (float)i/((float)LOG_LUT_SIZE);
		m_log_lut[i] = logf (log_res);
	}
    m_log_lut[LOG_LUT_SIZE] = 0.0f;

	// Frequently used values
	

	// Data term precomputation
	m_normalization_ground = new float[m_rows];
	m_inv_sigma2_ground = new float[m_rows];
	m_normalization_object = new float[m_max_dis];
	m_inv_sigma2_object = new float[m_max_dis];
	m_object_disparity_range = new float[m_max_dis];

	for(int i = 0; i < m_max_dis; i++) {
		float previous_mean = (float) i;
		m_object_disparity_range[i] = ComputeObjectDisparityRange(previous_mean);
	}

	// Precomputation of data term
	PrecomputeSky();
	PrecomputeObject();

	// Object Data Cost LUT
	m_obj_cost_lut = new float[m_max_dis*m_max_dis];

	for(int fn = 0; fn < m_max_dis; fn++) {
		for(int dis = 0; dis < m_max_dis; dis++) {
			m_obj_cost_lut[fn*m_max_dis+dis] = GetDataCostObject((float) fn, dis, (float) dis);
		}
	}

	const int rows_power2 = (int) powf(2, ceilf(log2f(m_rows+1)));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_big, m_rows*m_cols*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, m_rows*m_realcols*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_ground_function, m_rows*m_realcols*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_normalization_ground, m_rows*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inv_sigma2_ground, m_rows*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_object_disparity_range, m_max_dis*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_object_lut, rows_power2*m_realcols*m_max_dis*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_obj_cost_lut, m_max_dis*m_max_dis*sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(d_object_disparity_range, m_object_disparity_range, sizeof(float)*m_max_dis,
			cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_obj_cost_lut, m_obj_cost_lut, sizeof(float)*m_max_dis*m_max_dis,
			cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaStreamCreate(&m_stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&m_stream2));

	m_params.rows = m_rows;
	m_params.cols = m_realcols;
	m_params.max_dis = m_max_dis;
	m_params.rows_log = m_rows_log;
	m_params.normalization_sky = m_normalization_sky;
	m_params.inv_sigma2_sky = m_inv_sigma2_sky;
    m_params.exportProbabilitiesParameters = m_export_probabilities_params;
    m_params.cameraParameters = m_camera_parameters;
	m_params.range_objects_z = m_model_params.rangeObjectsZ;
    m_params.probabilitiesParameters = m_probabilities_params;
	m_params.epsilon = m_model_params.epsilon;
	m_params.rows_power2 = rows_power2;
	m_params.max_sections = m_max_sections;
	m_params.max_dis_log = m_max_dis_log;
	m_params.width_margin = m_model_params.widthMargin;
}

void Stixels::StixelsImpl::Finish() {
	delete[] m_cost_table;
	delete[] m_index_table;
	delete[] m_stixels;
	if(m_model_params.columnStep > 1) {
		CUDA_CHECK_RETURN(cudaFreeHost(m_disp_im_modified));
	}
	delete[] m_column;
	delete[] m_sum;
	delete[] m_valid;
	delete[] m_ground_function;
	delete[] m_normalization_ground;
	delete[] m_inv_sigma2_ground;
	delete[] m_normalization_object;
	delete[] m_inv_sigma2_object;
	delete[] m_object_disparity_range;
	delete[] m_obj_cost_lut;
	delete[] m_log_lut;

	CUDA_CHECK_RETURN(cudaFree(d_disparity_big));
	CUDA_CHECK_RETURN(cudaFree(d_disparity));
	CUDA_CHECK_RETURN(cudaFree(d_ground_function));
	CUDA_CHECK_RETURN(cudaFree(d_normalization_ground));
	CUDA_CHECK_RETURN(cudaFree(d_inv_sigma2_ground));
	CUDA_CHECK_RETURN(cudaFree(d_object_disparity_range));
	CUDA_CHECK_RETURN(cudaFree(d_object_lut));
	CUDA_CHECK_RETURN(cudaFree(d_stixels));
	CUDA_CHECK_RETURN(cudaFree(d_obj_cost_lut));

	CUDA_CHECK_RETURN(cudaStreamDestroy(m_stream1));
	CUDA_CHECK_RETURN(cudaStreamDestroy(m_stream2));
}

void Stixels::StixelsImpl::SetDisparityImage(pixel_t *disp_im) {
	m_disp_im = disp_im;

	// New image joining the columns
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_disparity_big, m_disp_im, sizeof(pixel_t)*m_rows*m_cols,
			cudaMemcpyHostToDevice, m_stream1));
}

void Stixels::StixelsImpl::SetProbabilities(ProbabilitiesParameters params) {
    m_probabilities_params = params;
    m_export_probabilities_params = ComputeProbabilitiesParameters(params, m_max_dis, m_rows);
}

ExportProbabilitiesParameters Stixels::StixelsImpl::ComputeProbabilitiesParameters(
    ProbabilitiesParameters params, int max_dist, int rows) {
    ExportProbabilitiesParameters ex_params;

    float pnexists_given_ground = (params.groundGivenNExist*params.nExistDis) / params.ground;
    float pnexists_given_object = (params.objectGivenNExist*params.nExistDis) / params.object;
    float pnexists_given_sky    = (params.skyGivenNExist   *params.nExistDis) / params.sky;

    m_max_dis_log = logf(m_max_disf);
    m_rows_log = logf((float)m_rows);

    ex_params.uniformSky = m_max_dis_log - logf(params.outSky);
    ex_params.uniform = m_max_dis_log - logf(params.out);
    ex_params.nExistsGivenSkyLOG = -logf(pnexists_given_sky);
    ex_params.nExistsGivenSkyNLOG = -logf(1.0f - pnexists_given_sky);
    ex_params.nExistsGivenGroundLOG = -logf(pnexists_given_ground);
    ex_params.nExistsGivenGroundNLOG = -logf(1.0f - pnexists_given_ground);
    ex_params.nExistsGivenObjectLOG = -logf(pnexists_given_object);
    ex_params.nExistsGivenObjectNLOG = -logf(1.0f - pnexists_given_object);

    return ex_params;
}

void Stixels::StixelsImpl::LoadDisparityImage(const pixel_t* m_disp, 
    cv::Size imageSize, EstimatedCameraParameters estimated_camera_params) {

}

void Stixels::StixelsImpl::LoadDisparityImageD(pixel_t* d_disp, 
    cv::Size imageSize, EstimatedCameraParameters estimated_camera_params) {

}

void Stixels::StixelsImpl::SetCameraParameters(CameraParameters camera_params, EstimatedCameraParameters estimated_camera_params) {
    m_camera_parameters = camera_params;
    m_estimated_camera_params = estimated_camera_params;
    m_estimated_camera_params.horizonPoint = m_rows - estimated_camera_params.horizonPoint - 1;
}

void Stixels::StixelsImpl::SetDisparityParameters(const int rows, const int cols, const int max_dis,
		const float sigma_disparity_object, const float sigma_disparity_ground, float sigma_sky) {
	m_rows = rows;
	m_cols = cols;
	m_max_dis = max_dis;
	m_max_disf = (float) m_max_dis;
    m_sigma_disparity_object = sigma_disparity_object;
    m_sigma_disparity_ground = sigma_disparity_ground;
	m_sigma_sky = sigma_sky;
}

void Stixels::StixelsImpl::SetModelParameters(StixelModelParameters stixel_model_parameters) {
    m_model_params = stixel_model_parameters;
}

float Stixels::StixelsImpl::Compute() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Precomputation of data term
	PrecomputeGround();

	CUDA_CHECK_RETURN(cudaMemcpy(d_ground_function, m_ground_function, sizeof(float)*m_rows,
			cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_normalization_ground, m_normalization_ground, sizeof(float)*m_rows,
			cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_inv_sigma2_ground, m_inv_sigma2_ground, sizeof(float)*m_rows,
			cudaMemcpyHostToDevice));

	JoinColumns<<<divUp(m_rows*m_realcols, 256), 256>>>(d_disparity_big, d_disparity, m_model_params.columnStep,
        m_model_params.medianStep, m_model_params.widthMargin, m_rows, m_cols, m_realcols);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
	    printf("Error: %s %d\n", cudaGetErrorString(err), err);
	}

	m_params.estimatedCameraParameters.horizonPoint = m_estimated_camera_params.horizonPoint;

	ComputeObjectLUT<<<m_realcols, 512>>>(d_disparity, d_obj_cost_lut, d_object_lut, m_params,
			(int) powf(2, ceilf(log2f(m_rows))));

	err = cudaGetLastError();
	if (err != cudaSuccess) {
	    printf("Error: %s %d\n", cudaGetErrorString(err), err);
	}

	int shared_mem_size = sizeof(float)*(m_params.rows_power2*6+m_params.max_dis)
			+sizeof(int16_t)*(m_params.rows_power2*3)+
			+sizeof(pixel_t)*(m_params.rows_power2);
#if ROBUST_MEAN_ESTIMATION
	shared_mem_size += sizeof(pixel_t)*(m_params.rows_power2*2);
#endif

	StixelsKernel<<<m_realcols, m_rows,
			shared_mem_size>>>(d_disparity, m_params, d_ground_function, d_normalization_ground, d_inv_sigma2_ground,
					d_object_disparity_range, d_object_lut, d_stixels);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
	    printf("Error: %s %d\n", cudaGetErrorString(err), err);
	}

	// Synchronize
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	cudaEventRecord(stop, 0);
	float elapsed_time_ms;
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	CUDA_CHECK_RETURN(cudaMemcpy(m_stixels, d_stixels, m_realcols*m_max_sections*sizeof(Section),
			cudaMemcpyDeviceToHost));
	return elapsed_time_ms;
}

float Stixels::StixelsImpl::FastLog(float v) {
	return m_log_lut[(int)((v)*LOG_LUT_SIZE+0.5f)];
}

pixel_t Stixels::StixelsImpl::ComputeMean(const int vB, const int vT, const int u) {
	const pixel_t valid_dif = m_valid[u*(m_rows+1)+vT+1]-m_valid[u*(m_rows+1)+vB];
	pixel_t mean = (valid_dif == 0) ? 0 : (m_sum[u*(m_rows+1)+vT+1]-m_sum[u*(m_rows+1)+vB])/valid_dif;

#if ROBUST_MEAN_ESTIMATION
		pixel_t total_weight = 0;
		pixel_t total_dv = 0;
		for(int v = vB; v <= vT; v++) {
			const pixel_t dv = m_column[u*m_rows+v];
			const int val = dv != INVALID_DISPARITY;
			const pixel_t weight = (pixel_t)val/(1+fabsf(dv-mean));
			total_weight += weight;
			total_dv += weight*dv;
		}
		mean = (total_weight == 0.0f) ? 0.0f : total_dv/total_weight;
#endif

	return mean;
}

void Stixels::StixelsImpl::PrecomputeGround() {
	const float fb = (m_camera_parameters.focal *m_camera_parameters.baseline)/ m_estimated_camera_params.cameraHeight;
	const float pout = m_probabilities_params.out;

	for(int v = 0; v < m_rows; v++) {
		const float fn = GroundFunction(v);
		m_ground_function[v] = fn;

		const float x = m_estimated_camera_params.pitch+(float)(m_estimated_camera_params.horizonPoint-v)/ m_camera_parameters.focal;
		const float sigma2_road = fb*fb*(m_estimated_camera_params.cameraHeight*m_estimated_camera_params.sigmaCameraHeight
				*x*x/(m_estimated_camera_params.cameraHeight*m_estimated_camera_params.cameraHeight)+ m_estimated_camera_params.sigmaCameraTilt*m_estimated_camera_params.sigmaCameraTilt);
        const float sigma = sqrtf(m_sigma_disparity_ground*m_sigma_disparity_ground+sigma2_road);

		const float a_range = 0.5f*(erf((m_max_disf-fn)/(sigma*sqrtf(2.0f)))-erf((-fn)/(sigma*sqrtf(2.0f))));

		m_normalization_ground[v] = FastLog(a_range) - FastLog((1.0f - pout)/(sigma*sqrtf(2.0f*PIFLOAT)));
		m_inv_sigma2_ground[v] = 1.0f/(2.0f*sigma*sigma);
	}
}

void Stixels::StixelsImpl::PrecomputeObject() {
	const float pout = m_probabilities_params.out;

	for(int dis = 0; dis < m_max_dis; dis++) {
		const float fn = (float) dis;

		const float sigma_object = fn*fn*m_model_params.rangeObjectsZ/(m_camera_parameters.focal *m_camera_parameters.baseline);
        const float sigma = sqrtf(m_sigma_disparity_object*m_sigma_disparity_object+sigma_object*sigma_object);

		const float a_range = 0.5f*(erf((m_max_disf-fn)/(sigma*sqrtf(2.0f)))-erf((-fn)/(sigma*sqrtf(2.0f))));

		m_normalization_object[dis] = FastLog(a_range) - FastLog((1.0f - pout)/(sigma*sqrtf(2.0f*PIFLOAT)));
		m_inv_sigma2_object[dis] = 1.0f/(2.0f*sigma*sigma);
	}
}

float Stixels::StixelsImpl::GetDataCostObject(const float fn, const int dis, const float d) {
    float data_cost = m_export_probabilities_params.nExistsGivenObjectLOG;
	if(!ALLOW_INVALID_DISPARITIES || d != INVALID_DISPARITY) {
		const float model_diff = (d-fn);
		const float pgaussian = m_normalization_object[dis] + model_diff*model_diff*m_inv_sigma2_object[dis];

		const float p_data = fminf(m_export_probabilities_params.uniform, pgaussian);
		data_cost = p_data + m_export_probabilities_params.nExistsGivenObjectNLOG;
	}
	return data_cost;
}

void Stixels::StixelsImpl::PrecomputeSky() {
	const float sigma = m_sigma_sky;
	const float pout = m_probabilities_params.outSky;

	const float a_range = 0.5f*(erf(m_max_disf/(sigma*sqrtf(2.0f)))-erf(0.0f));
	m_normalization_sky = FastLog(a_range) - logf((1.0f - pout)/(sigma*sqrtf(2.0f*PIFLOAT)));
	m_inv_sigma2_sky = 1.0f/(2.0f*sigma*sigma);
}

float Stixels::StixelsImpl::GroundFunction(const int v) {
	return m_estimated_camera_params.slope*(float)(m_estimated_camera_params.horizonPoint-v);
}

float Stixels::StixelsImpl::ComputeObjectDisparityRange(const float previous_mean) {
	float range_disp = 0.0f;
	if(previous_mean != 0) {
		const float pmean_plus_z = (m_camera_parameters.focal *m_camera_parameters.baseline /previous_mean) + m_model_params.rangeObjectsZ;
		range_disp = previous_mean - (m_camera_parameters.focal *m_camera_parameters.baseline /pmean_plus_z);
	}
	return range_disp;
}

void Stixels::StixelsImpl::malloc_memory() {

}

void Stixels::StixelsImpl::free_memory() {

}

void Stixels::StixelsImpl::malloc_image_memory() {

}

void Stixels::StixelsImpl::free_image_memory() {

}

template<typename T>
void Stixels::StixelsImpl::PrintTable(T *table) {
    for (int i = m_rows - 1; i >= 0; i--) {
        std::cout << i << "\t" << table[i * 3] << "\t" << table[i * 3 + 1]
            << "\t" << table[i * 3 + 2] << std::endl;
    }
}

void malloc_memory() {

}

void free_memory() {

}

void malloc_image_memory() {

}

void free_image_memory() {

}

Stixels::Stixels() : 
    m_impl(new StixelsImpl()) {}

Stixels::~Stixels() {}

void Stixels::Initialize() {
    m_impl->Initialize();
}

void Stixels::LoadDisparityImage(const pixel_t* m_disp, 
    cv::Size imageSize, EstimatedCameraParameters estimated_camera_params) {
    m_impl->LoadDisparityImage(m_disp, imageSize, estimated_camera_params);
}

void Stixels::LoadDisparityImageD(pixel_t* d_disp, 
    cv::Size imageSize, EstimatedCameraParameters estimated_camera_params) {
    m_impl->LoadDisparityImageD(d_disp, imageSize, estimated_camera_params);
}

float Stixels::Compute() {
    return m_impl->Compute();
}

Section* Stixels::GetStixels() {
    return m_impl->GetStixels();
}

int Stixels::GetRealCols() {
    return m_impl->GetRealCols();
}

int Stixels::GetMaxSections() {
    return m_impl->GetMaxSections();
}

void Stixels::Finish() {
    m_impl->Finish();
}

void Stixels::SetDisparityImage(pixel_t *disp_im) {
    m_impl->SetDisparityImage(disp_im);
}

void Stixels::SetProbabilities(ProbabilitiesParameters params) {
    m_impl->SetProbabilities(params);
}

void Stixels::SetCameraParameters(CameraParameters camera_params, EstimatedCameraParameters estimated_camera_params) {
    m_impl->SetCameraParameters(camera_params, estimated_camera_params);
}

void Stixels::SetDisparityParameters(const int rows, const int cols, const int max_dis,
    const float sigma_disparity_object, const float sigma_disparity_ground, float sigma_sky) {
    m_impl->SetDisparityParameters(rows, cols, max_dis,
        sigma_disparity_object, sigma_disparity_ground, sigma_sky);
}

void Stixels::SetModelParameters(StixelModelParameters stixel_model_parameters) {
    m_impl->SetModelParameters(stixel_model_parameters);
}