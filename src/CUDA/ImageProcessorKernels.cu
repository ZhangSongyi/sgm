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

#include "ImageProcessorKernels.cuh"
#include "configuration.h"

__global__ void MedianFilter3x3(const uint8_t* __restrict__ d_input, uint8_t* __restrict__ d_out, const uint32_t rows, const uint32_t cols) {
	MedianFilter<9>(d_input, d_out, rows, cols);
}

__global__ void ColorizeDisparityImage(const uint8_t* __restrict__ d_input, uint8_t* __restrict__ d_out, uint8_t* __restrict__ d_color_table, const uint32_t rows, const uint32_t cols)
{
    const uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    const uint32_t row = idx / cols;
    const uint32_t col = idx % cols;

    if (row < rows && col < cols) {
        const int disp = d_input[idx];
        d_out[idx * 3] = d_color_table[disp * 3];
        d_out[idx * 3 + 1] = d_color_table[disp * 3 + 1];
        d_out[idx * 3 + 2] = d_color_table[disp * 3 + 2];
    }
}