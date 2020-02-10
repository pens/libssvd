#include "svd.h"
#include <algorithm>
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_CPP
#include <lapacke.h>
using namespace std;
using namespace chrono;
using namespace ssvd;

// TODO return -1 on bad

// HACK s_n needed as MAGMA can't calculate only k eigenvalues
void eigsToSingVals(float *sigma, int s_n, float *v, int n, int k) {
  for (auto i = 0; i < s_n / 2; ++i)
    swap(sigma[i], sigma[s_n - 1 - i]);
  for (auto i = 0; i < k; ++i)
    sigma[i] = sqrt(abs(sigma[i]));
  for (auto i = 0; i < k / 2; ++i)
    for (auto j = 0; j < n; ++j)
      swap(v[i * n + j], v[(k - 1 - i) * n + j]);
}

StreamingSvdCpu::StreamingSvdCpu(int m, int n, int k)
    : m(m), n(n), k(k), xx(n * n), xx_temp(n * n), isuppz(2 * n), work(),
      iwork() {
  float sz_work;
  int sz_iwork;
  OK(LAPACKE_ssyevr_work(LAPACK_COL_MAJOR, 'V', 'I', 'U', n, nullptr, n, 0.0f,
                         0.0f, n - k - 1, n, 0.0f, nullptr, nullptr, nullptr, n,
                         nullptr, &sz_work, -1, &sz_iwork, -1));
  work.resize(sz_work);
  iwork.resize(sz_iwork);
}

int StreamingSvdCpu::Run(const float *x, int x_n, bool stream, float *sigma,
                         float *v, double *elapsed) {
  auto start = high_resolution_clock::now();

  // Build the matrix X.T*X
  if (!stream) {
    // Calculate X.T * X
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, m, 1.0f, x, m, x,
                m, 0.0f, xx.data(), n);
  } else {
    /*
        Shift X.T*X up and left x_n rows / columns
        [a b c]    [d e -]
        [b d e] -> [e f -]
        [c e f]    [- - -] x_n
                        x_n
    */
    memmove(xx.data(), xx.data() + n * x_n + x_n,
            (n * (n - x_n) - x_n) * sizeof(float));

    /*
        Calculate new column(s) of X.T*X
        [d e -]    [d e g]
        [e f -] -> [e f h]
        [- - -]    [- - i]
    */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, x_n, m, 1.0f, x, m,
                x + m * (n - x_n), m, 0.0f, xx.data() + n * (n - x_n), n);

    /*
        Copy last column into last row (X.T*X is symmetric)
        [d e g]    [d e g]
        [e f h] -> [e f h]
        [- - i]    [g h i]
    */
    for (auto i = 0; i < n - 2; ++i)
      xx[(i + 1) * n - 1] = xx[n * (n - 1) + i];
  }

  // Find the eigenvalues and eigenvectors of X.T*X
  // Eigenvectors of X.T*X = Right-singular vectors V
  // LAPACK claims ssyevr to be the fastest eigensolver
  int eig_count;
  xx_temp = xx;
  OK(LAPACKE_ssyevr_work(LAPACK_COL_MAJOR, 'V', 'I', 'U', n, xx_temp.data(), n,
                         0.0f, 0.0f, n - k + 1, n, 0.0f, &eig_count, sigma, v,
                         n, isuppz.data(), work.data(), work.size(),
                         iwork.data(), iwork.size()));

  // Square roots of eigenvalues of X.T*X = Singular values Sigma
  eigsToSingVals(sigma, k, v, n, k);

  if (elapsed)
    *elapsed =
        duration_cast<duration<double>>(high_resolution_clock::now() - start)
            .count();

  return 0;
}

StreamingSvdGpu::StreamingSvdGpu(int m, int n, int k, int n_full)
    : m(m), n(n), k(k), n_full(n_full <= 0 ? n : n_full), wA(n * n),
      iwork(3 + 5 * n), sigma_temp(n) {
  MAGMA_OK(magma_init());

  magma_queue_create(0, &queue);

  MAGMA_OK(magma_smalloc(&dX, m * this->n_full));
  MAGMA_OK(magma_smalloc(&dXX, n * n));
  MAGMA_OK(magma_smalloc(&dXX_dV, n * n));

  float sz_work;
  magma_int_t sz_iwork;
  magma_int_t info;
  MAGMA_OK(magma_ssyevdx_gpu(MagmaVec, MagmaRangeI, MagmaUpper, n, nullptr, n,
                             0.0f, 0.0f, n - k + 1, n, nullptr, nullptr,
                             nullptr, n, &sz_work, -1, &sz_iwork, -1, &info));
  work.resize(sz_work);
  iwork.resize(sz_iwork);
}

StreamingSvdGpu::~StreamingSvdGpu() {
  MAGMA_OK(magma_free(dXX_dV));
  MAGMA_OK(magma_free(dXX));
  MAGMA_OK(magma_free(dX));

  magma_queue_destroy(queue);

  MAGMA_OK(magma_finalize());
}

int StreamingSvdGpu::Run(const float *x, int x_n, bool stream, float *sigma,
                         float *v, double *elapsed) {
  real_Double_t time = 0.0;
  if (!stream) {
    magma_ssetmatrix(m, n_full, x, m, dX, m, queue);

    time -= magma_sync_wtime(queue);

    magma_sgemm(MagmaTrans, MagmaNoTrans, n, n, m, 1.0f, dX, m, dX, m, 0.0f,
                dXX, n, queue);

    time += magma_sync_wtime(queue);
  } else {
    for (auto i = 0; i < n_full - x_n; ++i)
      magma_scopymatrix(m, 1, dX + (i + 1) * m, m, dX + i * m, m, queue);
    magma_ssetmatrix(m, x_n, x + (n_full - x_n) * m, m, dX + (n_full - x_n) * m,
                     m, queue);

    time -= magma_sync_wtime(queue);

    magma_scopyvector(n * (n - x_n) - x_n, dXX + n * x_n + x_n, 1, dXX, 1,
                      queue);
    magma_sgemm(MagmaTrans, MagmaNoTrans, n, x_n, m, 1.0f, dX, m,
                dX + m * (n - x_n), m, 0.0f, dXX + n * (n - x_n), n, queue);
    for (auto i = 1; i <= x_n; ++i)
      magma_scopyvector(n - i, dXX + n * (n - i), 1, dXX + n - i, n, queue);

    time += magma_sync_wtime(queue);
  }

  magma_scopymatrix(n, n, dXX, n, dXX_dV, n, queue);

  time -= magma_sync_wtime(queue);

  magma_int_t info;
  int mout;
  // MAGMA has not yet implemented ssyevr
  MAGMA_OK(magma_ssyevdx_gpu(MagmaVec, MagmaRangeI, MagmaUpper, n, dXX_dV, n,
                             0.f, 0.f, n - k + 1, n, &mout, sigma_temp.data(),
                             wA.data(), n, work.data(), work.size(),
                             iwork.data(), iwork.size(), &info));

  time += magma_sync_wtime(queue);

  magma_sgetmatrix(n, k, dXX_dV + (n - k) * n, n, v, n, queue);

  auto time2 = high_resolution_clock::now();

  eigsToSingVals(sigma_temp.data() + n - k, k, v, n, k);
  memcpy(sigma, sigma_temp.data() + n - k, k * 4);

  time += duration_cast<duration<double>>(high_resolution_clock::now() - time2)
              .count();

  if (elapsed)
    *elapsed = time;

  return 0;
}
