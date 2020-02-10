#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run Singular Value Decomposition using method of snapshots.
 *
 * If streaming is true, call SvdUpdate() to run streaming update.
 * Call SvdStop() to finish streaming.
 *
 * x: m x n
 * sigma: k
 * v: n x k
 */
int Svd(const float *x, int m, int n, int k, bool gpu, bool streaming,
        float *sigma, float *v);
int SvdUpdate(const float *x_new, int n_new, float *sigma, float *v);
void SvdStop();

/**
 * Run Dynamic Mode Decomposition.
 *
 * If streaming is true, call DmdUpdate() to run streaming update.
 * Call DmdStop() to finish streaming.
 *
 * x: m x n
 * lambda: k * 2 (complex)
 * phi: m x k (packed complex)
 */
int Dmd(const float *x, int m, int n, int k, bool gpu, bool streaming,
        float *lambda, float *phi);
int DmdUpdate(const float *x_new, int n_new, float *lambda, float *phi);
void DmdStop();

/**
 * Get elapsed time for computations (not memory transfers).
 */
double GetElapsedCalcs();

#ifdef __cplusplus
};
#endif
