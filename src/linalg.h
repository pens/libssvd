/*
    Copyright (c) 2016-7 Seth Pendergrass. See LICENSE.
*/
#pragma once
#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <complex>
#define ADD_
#define HAVE_LAPACK_CONFIG_H_
#define LAPACK_COMPLEX_STRUCTURE
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>