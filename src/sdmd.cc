/*
    Copyright (c) 2016-7 Seth Pendergrass. See LICENSE.
*/
#include "sdmd.h"
#include <chrono>
#include <cstring>
#include "linalg.h"
using namespace std;
using namespace chrono;
using namespace ssvd;

SdmdCpu::SdmdCpu(int m, int n)
    : m(m),
      n(n),
      k(n - 1),
      svd(m, n - 1),
      sigma(k),
      v(k * k),
      w(k * k),
      lambda_real(k),
      lambda_imag(k),
      phi(m * k),
      xy(k * k),
      vs(k * k),
      uy(k * k),
      a(k * k),
      vsw(k * k),
      sigma_inv_mat(k * k) {}

int SdmdCpu::Run(const float *x, int x_n, bool stream, float *lambda,
                 float *phi, double *elapsed, int k_sigma) {
  int res = svd.Run(x, x_n, stream, sigma.data(), v.data(), elapsed);
  if (res) return -1;

  auto start = high_resolution_clock::now();
  if (!stream) {
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, k, m, 1.0f, x, m,
                x + m, m, 0.0f, xy.data(), k);
  } else {
    memmove(xy.data(), svd.GetXX() + k, k * (k - 1) * sizeof(float));
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, 1, m, 1.0f, x, m,
                x + m * k, m, 0.0f, xy.data() + k * (k - 1), k);
  }

  auto num_sigma = (k_sigma > 0) ? min(k, k_sigma) : k;
  for (auto i = 0; i < num_sigma; ++i) sigma_inv_mat[i * n] = 1.0f / sigma[i];

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, k, k, 1.0f,
              v.data(), k, sigma_inv_mat.data(), k, 0.0f, vs.data(), k);
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, k, k, 1.0f, vs.data(),
              k, xy.data(), k, 0.0f, uy.data(), k);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, k, k, 1.0f,
              uy.data(), k, vs.data(), k, 0.0f, a.data(), k);

  res = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', k, a.data(), k,
                      lambda_real.data(), lambda_imag.data(), nullptr, 1,
                      w.data(), k);
  if (res) return -1;

  auto time = high_resolution_clock::now() - start;
  for (int i = 0; i < k; ++i) {
    lambda[2 * i] = lambda_real[i];
    lambda[2 * i + 1] = lambda_imag[i];
  }
  start = high_resolution_clock::now();

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, k, k, 1.0f,
              vs.data(), k, w.data(), k, 0.0f, vsw.data(), k);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, k, 1.0f, &x[m],
              m, vsw.data(), k, 0.0f, phi, m);
  if (elapsed)
    *elapsed += duration_cast<duration<double>>(high_resolution_clock::now() -
                                                start + time)
                    .count();

  return 0;
}

SdmdMagma::SdmdMagma(int m, int n)
    : m(m),
      n(n),
      k(n - 1),
      svd(m, k, n),
      sigma(k),
      v(k * k),
      w(k * k),
      lambda_real(k),
      lambda_imag(k),
      a(k * k),
      sigma_inv(k),
      queue(svd.GetQueue()),
      dX(svd.GetDX()) {
  magma_smalloc(&dXY, k * k);
  magma_smalloc(&dSigma_inv, k * k);
  magma_smalloc(&dVS, k * k);
  magma_smalloc(&dUY, k * k);
  magma_smalloc(&dA, k * k);
  magma_smalloc(&dW, k * k);
  magma_smalloc(&dVSW, k * k);
  magma_smalloc(&dPhi, m * k);
  magma_smalloc(&dV, k * k);
}

SdmdMagma::~SdmdMagma() {
  magma_free(dV);
  magma_free(dPhi);
  magma_free(dVSW);
  magma_free(dW);
  magma_free(dA);
  magma_free(dUY);
  magma_free(dVS);
  magma_free(dSigma_inv);
  magma_free(dXY);
}

int SdmdMagma::Run(const float *x, int x_n, bool stream, float *lambda,
                   float *phi, double *elapsed, int k_sigma) {
  int res = svd.Run(x, x_n, stream, sigma.data(), v.data(), elapsed);
  if (res) return -1;

  real_Double_t time = 0.0;
  time -= magma_sync_wtime(queue);
  magma_scopyvector(k * (k - 1), svd.GetDXX() + k, 1, dXY, 1, queue);
  magma_sgemm(MagmaTrans, MagmaNoTrans, k, 1, m, 1.0f, dX, m, dX + m * k, m,
              0.0f, dXY + k * (k - 1), k, queue);
  time += magma_sync_wtime(queue);

  auto time_cpu = high_resolution_clock::now();
  auto num_sigma = (k_sigma > 0) ? min(k, k_sigma) : k;
  for (auto i = 0; i < num_sigma; ++i) sigma_inv[i] = 1.0f / sigma[i];
  time +=
      duration_cast<duration<double>>(high_resolution_clock::now() - time_cpu)
          .count();

  magma_ssetmatrix(k, k, v.data(), k, dV, k, queue);
  magmablas_slaset(MagmaFull, k, k, 0.0f, 0.0f, dSigma_inv, k, queue);
  magma_ssetvector(k, sigma_inv.data(), 1, dSigma_inv, k + 1, queue);

  time -= magma_sync_wtime(queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, k, k, k, 1., dV, k, dSigma_inv, k, 0.,
              dVS, k, queue);
  magma_sgemm(MagmaTrans, MagmaNoTrans, k, k, k, 1., dVS, k, dXY, k, 0., dUY, k,
              queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, k, k, k, 1., dUY, k, dVS, k, 0., dA,
              k, queue);
  time += magma_sync_wtime(queue);

  magma_sgetmatrix(k, k, dA, k, a.data(), k, queue);

  time_cpu = high_resolution_clock::now();
  // MAGMA version causes OpenMP warnings on Windows. Will use CPU at this
  // size anyway
  res = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', k, a.data(), k,
                      lambda_real.data(), lambda_imag.data(), nullptr, 1,
                      w.data(), k);
  time +=
      duration_cast<duration<double>>(high_resolution_clock::now() - time_cpu)
          .count();
  if (res) return -1;

  for (int i = 0; i < k; ++i) {
    lambda[2 * i] = lambda_real[i];
    lambda[2 * i + 1] = lambda_imag[i];
  }

  magma_ssetmatrix(k, k, w.data(), k, dW, k, queue);

  time -= magma_sync_wtime(queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, k, k, k, 1.0f, dVS, k, dW, k, 0.0f,
              dVSW, k, queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, m, k, k, 1.0f, dX + m, m, dVSW, k,
              0.0f, dPhi, m, queue);
  time += magma_sync_wtime(queue);
  if (phi) magma_sgetmatrix(m, k, dPhi, m, phi, m, queue);

  if (elapsed) *elapsed += time;
  return 0;
}
