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

SdmdCpu::SdmdCpu(int m, int n, int k)
    : m(m),
      n(n),
      k(k),
      svd(m, n - 1, k),
      sigma(k),
      v((n - 1) * k),
      w(k * k),
      lambda_real(k),
      lambda_imag(k),
      phi(m * k),
      xy((n - 1) * (n - 1)),
      vs((n - 1) * k),
      uy(k * (n - 1)),
      a(k * k),
      vsw((n - 1) * k),
      sigma_inv_mat(k * k) {}

int SdmdCpu::Run(const float *x, int x_n, bool stream, float *lambda,
                 float *phi, double *elapsed) {
  int res = svd.Run(x, x_n, stream, sigma.data(), v.data(), elapsed);
  if (res) return -1;

  auto start = high_resolution_clock::now();
    memmove(xy.data(), svd.GetXX() + n - 1, (n - 1) * (n - 1 - 1) * sizeof(float));
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n - 1, 1, m, 1.0f, x, m, x + m * (n - 1), m, 0.0f, xy.data() + (n - 1) * (n - 1 - 1), n - 1);

  for (auto i = 0; i < k; ++i) sigma_inv_mat[i * (k + 1)] = 1.0f / sigma[i];

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n - 1, k, k, 1.0f, v.data(), n - 1, sigma_inv_mat.data(), k, 0.0f, vs.data(), n - 1);
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, n - 1, n - 1, 1.0f, vs.data(), n - 1, xy.data(), n - 1, 0.0f, uy.data(), k);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, k, n - 1, 1.0f, uy.data(), k, vs.data(), n - 1, 0.0f, a.data(), k);

  res = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', k, a.data(), k, lambda_real.data(), lambda_imag.data(), nullptr, 1, w.data(), k);
  if (res) return -1;

  auto time = high_resolution_clock::now() - start;
  for (int i = 0; i < k; ++i) {
    lambda[2 * i] = lambda_real[i];
    lambda[2 * i + 1] = lambda_imag[i];
  }
  start = high_resolution_clock::now();

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n - 1, k, k, 1.0f, vs.data(), n - 1, w.data(), k, 0.0f, vsw.data(), n - 1);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, n - 1, 1.0f, &x[m], m, vsw.data(), n - 1, 0.0f, phi, m);
  if (elapsed)
    *elapsed += duration_cast<duration<double>>(high_resolution_clock::now() - start + time).count();

  return 0;
}

SdmdMagma::SdmdMagma(int m, int n, int k)
    : m(m),
      n(n),
      k(k),
      svd(m, n - 1, k, n),
      sigma(k),
      v((n - 1) * k),
      w(k * k),
      lambda_real(k),
      lambda_imag(k),
      a(k * k),
      sigma_inv(k),
      queue(svd.GetQueue()),
      dX(svd.GetDX()) {
  magma_smalloc(&dXY, (n - 1) * (n - 1));
  magma_smalloc(&dSigma_inv, k * k);
  magma_smalloc(&dVS, (n - 1) * k);
  magma_smalloc(&dUY, k * (n - 1));
  magma_smalloc(&dA, k * k);
  magma_smalloc(&dW, k * k);
  magma_smalloc(&dVSW, (n - 1) * k);
  magma_smalloc(&dPhi, m * k);
  //TODO reuse from SVD?
  magma_smalloc(&dV, (n - 1) * k);
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
                   float *phi, double *elapsed) {
  int res = svd.Run(x, x_n, stream, sigma.data(), v.data(), elapsed);
  if (res) return -1;

  real_Double_t time = 0.0;
  time -= magma_sync_wtime(queue);
  magma_scopyvector((n - 1) * (n - 1 - 1), svd.GetDXX() + (n - 1), 1, dXY, 1, queue);
  magma_sgemm(MagmaTrans, MagmaNoTrans, n - 1, 1, m, 1.0f, dX, m, dX + m * (n - 1), m, 0.0f, dXY + (n - 1) * (n - 1 - 1), n - 1, queue);
  time += magma_sync_wtime(queue);

  auto time_cpu = high_resolution_clock::now();

  for (auto i = 0; i < k; ++i)
      sigma_inv[i] = 1.0f / sigma[i];

  time += duration_cast<duration<double>>(high_resolution_clock::now() - time_cpu).count();

  magma_ssetmatrix(n - 1, k, v.data(), n - 1, dV, n - 1, queue);
  magmablas_slaset(MagmaFull, k, k, 0.0f, 0.0f, dSigma_inv, k, queue);
  magma_ssetvector(k, sigma_inv.data(), 1, dSigma_inv, k + 1, queue);

  time -= magma_sync_wtime(queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, n - 1, k, k, 1., dV, n - 1, dSigma_inv, k, 0., dVS, n - 1, queue);
  magma_sgemm(MagmaTrans, MagmaNoTrans, k, n - 1, n - 1, 1., dVS, n - 1, dXY, n - 1, 0., dUY, k, queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, k, k, n - 1, 1., dUY, k, dVS, n - 1, 0., dA, k, queue);
  time += magma_sync_wtime(queue);

  magma_sgetmatrix(k, k, dA, k, a.data(), k, queue);

  time_cpu = high_resolution_clock::now();
  res = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', k, a.data(), k, lambda_real.data(), lambda_imag.data(), nullptr, 1, w.data(), k);
  time += duration_cast<duration<double>>(high_resolution_clock::now() - time_cpu).count();
  if (res) return -1;

  for (int i = 0; i < k; ++i) {
    lambda[2 * i] = lambda_real[i];
    lambda[2 * i + 1] = lambda_imag[i];
  }

  magma_ssetmatrix(k, k, w.data(), k, dW, k, queue);

  time -= magma_sync_wtime(queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, n - 1, k, k, 1.0f, dVS, n - 1, dW, k, 0.0f, dVSW, n - 1, queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, m, k, n - 1, 1.0f, dX + m, m, dVSW, n - 1, 0.0f, dPhi, m, queue);
  time += magma_sync_wtime(queue);
  if (phi) magma_sgetmatrix(m, k, dPhi, m, phi, m, queue);

  if (elapsed) *elapsed += time;
  return 0;
}
