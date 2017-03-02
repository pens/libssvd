/**
\file libssvd.h
\copyright Copyright (c) 2016-7 Seth Pendergrass. See LICENSE.
*/
#pragma once
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _WIN32
#ifdef EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
Calculate the (partial) Singular Value Decomposition of X:
\f$X=U\Sigma V^*\f$
\param [in] x m x n input matrix
\param [in] m Rows in x
\param [in] n Columns in x
\param [in] gpu If true, use GPU acceleration
\param [in] streaming If true, enable streaming. Once enabled, use SVDUpdate()
to perform a streaming update. SVDStop() must be called to free resources
\param [out] sigma n length array of singular values, corresponding to the
diagonal of matrix \f$\Sigma\f$
\param [out] v n x n matrix of right-singular values
\return If non-zero, the function has failed
*/
API int SVD(const float *x, int m, int n, bool gpu, bool streaming,
            float *sigma, float *v);

/**
Updates the streaming Singular Value Decomposition. SVD() must have been called
with streaming == true.
\param [in] x_new New start of x after shifted n_new
\param [in] n_new Number of columns shifted
\param [out] sigma n length array of singular values
\param [out] v n x n matrix of right-singular values
\return If non-zero, the function has failed
*/
API int SVDUpdate(const float *x_new, int n_new, float *sigma, float *v);

/**
Must be called to free resources after finished if streaming == true in
SVD().
*/
API void SVDStop();

/**
Calculate the Dynamic Mode Decomposition of X:
\f$X\rightarrow\Phi,\Lambda\f$
\param [in] x m x n input matrix
\param [in] m Rows in x
\param [in] n Columns in x
\param [in] gpu If true, use GPU acceleration
\param [in] streaming If true, enable streaming. Once enabled, use DMDUpdate()
to perform a streaming update. DMDStop() must be called to free resources
\param [out] lambda 2 * n-1 length array of DMD eigenvalues. The i-th
eigenvalue has its real part in lambda[2 * i] and imaginary in
lambda[2 * i + 1
\param [out] phi m x n-1 matrix of DMD modes. phi is returned in packed form,
where the unpacked i-th column is (phi[:, i], phi[:, i+1) if real(lambda[i]) =
real(lambda[i+1]), (phi[:, i-1], -phi[:, i]) if real(lambda[i-1]) =
real(lambda[i]), else (phi[:, i], 0)
\return If non-zero, the function has failed
*/
API int DMD(const float *x, int m, int n, bool gpu, bool streaming,
            float *lambda, float *phi);

/**
Updates the streaming Dynamic Mode Decomposition. DMD() must have been called
with streaming == true.
\param [in] x_new New start of x after shifted n_new
\param [in] n_new Number of columns shifted
\param [out] lambda 2 * (n-1) length array of DMD eigenvalues. The i-th
eigenvalue has its real part in lambda[2 * i] and imaginary in
lambda[2 * i + 1]
\param [out] phi m x n-1 matrix of DMD modes. phi is returned in packed form,
where the unpacked i-th column is (phi[:, i], phi[:, i+1) if real(lambda[i]) =
real(lambda[i+1]), (phi[:, i-1], -phi[:, i]) if real(lambda[i-1]) =
real(lambda[i]), else (phi[:, i], 0)
\return If non-zero, the function has failed
*/
API int DMDUpdate(const float *x_new, int n_new, float *lambda, float *phi);

/**
Must be called to free resources after finished if streaming == true in
DMD().
*/
API void DMDStop();

/**
Returns the benchmarking time for the last function called
\param [out] time Elapsed time
*/
API void GetElapsed(double *time);

#ifdef __cplusplus
};
#endif
