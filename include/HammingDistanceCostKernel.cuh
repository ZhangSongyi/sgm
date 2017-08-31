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

#ifndef HAMMING_DISTANCE_COST_KERNEL_CUH_
#define HAMMING_DISTANCE_COST_KERNEL_CUH_

#include "configuration.h"
#include "util.hpp"
#include <stdint.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void HammingDistanceCostKernel (  const cost_t *d_transform0, const cost_t *d_transform1,
        uint8_t *d_cost, const int rows, const int cols );

#endif /* HAMMING_DISTANCE_COST_KERNEL_CUH_ */
