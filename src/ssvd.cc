/*
    Copyright (c) 2016-7 Seth Pendergrass. See LICENSE.
*/
#include "ssvd.h"
#include <chrono>
#include <cstring>
#include "linalg.h"
using namespace std;
using namespace chrono;
using namespace ssvd;

SsvdCpu::SsvdCpu(int m, int n, int k)
    : m(m), n(n), xx(n * n), v(n * n), sigma(n), isuppz(2 * n) {}

int SsvdCpu::Run(const float *x, int x_n, bool stream, float *sigma, float *v,
                 double *elapsed) {
  auto start = high_resolution_clock::now();

  if (!stream) {
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, m, 1.0f, x, m, x,
                m, 0.0f, xx.data(), n);
  } else {
    memmove(xx.data(), xx.data() + n * x_n + x_n,
            (n * (n - x_n) - x_n) * sizeof(float));
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, x_n, m, 1.0f, x, m,
                x + m * (n - x_n), m, 0.0f, xx.data() + n * (n - x_n), n);
    for (auto i = 0; i < n - 1; ++i) xx[(i + 1) * n - 1] = xx[n * (n - 1) + i];
  }
  int eig_count;
  this->v = xx;
  vector<float> v2(n * n);
  int res = LAPACKE_ssyevr(LAPACK_COL_MAJOR, 'V', 'A', 'U', n, this->v.data(),
                           n, 0.0f, 0.0f, 0, 0, 0.0f, &eig_count,
                           this->sigma.data(), v2.data(), n, isuppz.data());
  for (auto i = 0; i < n; ++i)
    for (auto j = 0; j < n; ++j) v[i * n + j] = v2[(n - 1 - i) * n + j];
  if (res) return -1;

  for (auto i = 0; i < n; ++i) sigma[i] = sqrt(abs(this->sigma[n - 1 - i]));

  if (elapsed)
    *elapsed =
        duration_cast<duration<double>>(high_resolution_clock::now() - start)
            .count();
  return 0;
}

SsvdMagma::SsvdMagma(int m, int n, int k, int n_full)
    : m(m), n(n), n_full(n_full), sigma(n), wA(n * n) {
  magma_init();
  magma_queue_create(0, &queue);
  if (n_full <= 0) this->n_full = n;
  magma_smalloc(&dX, m * this->n_full);
  magma_smalloc(&dXX, n * n);
  magma_smalloc(&dV, n * n);
  work.resize(max(2 * n + n * magma_get_ssytrd_nb(n), 1 + 6 * n + 2 * n * n));
  iwork.resize(3 * 5 * n);
}

SsvdMagma::~SsvdMagma() {
  magma_free(dV);
  magma_free(dXX);
  magma_free(dX);
  magma_queue_destroy(queue);
  magma_finalize();
}

int SsvdMagma::Run(const float *x, int x_n, bool stream, float *sigma, float *v,
                   double *elapsed) {
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

  magma_scopymatrix(n, n, dXX, n, dV, n, queue);

  magma_int_t info = 0;
  time -= magma_sync_wtime(queue);
  magma_ssyevd_gpu(MagmaVec, MagmaUpper, n, dV, n, this->sigma.data(),
                   wA.data(), n, work.data(), static_cast<int>(work.size()),
                   iwork.data(), static_cast<int>(iwork.size()), &info);
  time += magma_sync_wtime(queue);
  if (info) return -1;

  auto time2 = high_resolution_clock::now();
  for (auto i = 0; i < n; ++i) sigma[i] = sqrt(abs(this->sigma[n - 1 - i]));
  time += duration_cast<duration<double>>(high_resolution_clock::now() - time2)
              .count();

  magma_sgetmatrix(n, n, dV, n, v, n, queue);
  for (auto i = 0; i < n / 2; ++i)
    for (auto j = 0; j < n; ++j) swap(v[i * n + j], v[(n - 1 - i) * n + j]);

  if (elapsed) *elapsed = time;
  return 0;
}
