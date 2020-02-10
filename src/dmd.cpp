#include "dmd.h"
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

void invertSigma(const vector<float> &sigma, int k, int stride,
                 vector<float> &sigma_inv) {
  for (auto i = 0; i < k; ++i) {
    sigma_inv[i * stride] = sigma[i] != 0.0f ? 1.0f / sigma[i] : 0.0f;
  }
}

void mergeLambda(const vector<float> &real, const vector<float> &imag, int k,
                 float *lambda) {
  for (auto i = 0; i < k; ++i) {
    lambda[2 * i] = real[i];
    lambda[2 * i + 1] = imag[i];
  }
}

StreamingDmdCpu::StreamingDmdCpu(int m, int n, int k)
    : m(m), n(n), k(k), svd(m, n - 1, k), sigma(k + 1), v((n - 1) * k),
      w(k * k), lambda_real(k), lambda_imag(k), phi(m * k),
      xy((n - 1) * (n - 1)), vs((n - 1) * k), uy(k * (n - 1)), a(k * k),
      vsw((n - 1) * k), sigma_inv_mat(k * k) {
  float sz_work;
  OK(LAPACKE_sgeev_work(LAPACK_COL_MAJOR, 'N', 'V', k, nullptr, k, nullptr,
                        nullptr, nullptr, 1, nullptr, k, &sz_work, -1));
  work.resize(sz_work);
}

int StreamingDmdCpu::Run(const float *x, int x_n, bool stream, float *lambda,
                         float *phi, double *elapsed) {
  OK(svd.Run(x, x_n, stream, sigma.data(), v.data(), elapsed));

  auto start = high_resolution_clock::now();

  /*
      Shift one column over to convert X.T*X to X.T*Y
      [a b c]    [b c -]
      [b d e] -> [d e -]
      [c e f]    [e f -]
  */
  memmove(xy.data(), svd.GetXX() + n - 1,
          (n - 1) * (n - 1 - 1) * sizeof(float));

  /*
      Calculate last column of X.T*Y
      [a b c]    [b c g]
      [b d e] -> [d e h]
      [c e f]    [e f i]
  */
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n - 1, 1, m, 1.0f, x, m,
              x + m * (n - 1), m, 0.0f, xy.data() + (n - 1) * (n - 1 - 1),
              n - 1);

  // V * inv(Sigma)
  invertSigma(sigma, k, k + 1, sigma_inv_mat);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n - 1, k, k, 1.0f,
              v.data(), n - 1, sigma_inv_mat.data(), k, 0.0f, vs.data(), n - 1);

  // U*Y = V*inv(Sigma) * X.T*Y
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, n - 1, n - 1, 1.0f,
              vs.data(), n - 1, xy.data(), n - 1, 0.0f, uy.data(), k);

  // A_tilde = U*Y * V*inv(Sigma)
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, k, n - 1, 1.0f,
              uy.data(), k, vs.data(), n - 1, 0.0f, a.data(), k);

  // W, lambda = eig(A_tilde)
  // SGEEVX could improve conditioning of eigs
  OK(LAPACKE_sgeev_work(LAPACK_COL_MAJOR, 'N', 'V', k, a.data(), k,
                        lambda_real.data(), lambda_imag.data(), nullptr, 1,
                        w.data(), k, work.data(), work.size()));

  auto time = high_resolution_clock::now() - start;

  mergeLambda(lambda_real, lambda_imag, k, lambda);

  start = high_resolution_clock::now();

  // V*inv(Sigma) * W
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n - 1, k, k, 1.0f,
              vs.data(), n - 1, w.data(), k, 0.0f, vsw.data(), n - 1);

  // Phi = Y * V*inv(Sigma)*W
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, n - 1, 1.0f,
              &x[m], m, vsw.data(), n - 1, 0.0f, phi, m);

  if (elapsed)
    *elapsed += duration_cast<duration<double>>(high_resolution_clock::now() -
                                                start + time)
                    .count();

  return 0;
}

StreamingDmdGpu::StreamingDmdGpu(int m, int n, int k)
    : m(m), n(n), k(k), svd(m, n - 1, k, n), sigma(n - 1), v((n - 1) * k),
      w(k * k), lambda_real(k), lambda_imag(k), a(k * k), sigma_inv(k),
      queue(svd.GetQueue()), dX(svd.GetDX()) {
  MAGMA_OK(magma_smalloc(&dXY, (n - 1) * (n - 1)));
  MAGMA_OK(magma_smalloc(&dSigma_inv, k * k));
  MAGMA_OK(magma_smalloc(&dVS, (n - 1) * k));
  MAGMA_OK(magma_smalloc(&dUY, k * (n - 1)));
  MAGMA_OK(magma_smalloc(&dA, k * k));
  MAGMA_OK(magma_smalloc(&dW, k * k));
  MAGMA_OK(magma_smalloc(&dVSW, (n - 1) * k));
  MAGMA_OK(magma_smalloc(&dPhi, m * k));
  // TODO could reuse buffer from SVD
  MAGMA_OK(magma_smalloc(&dV, (n - 1) * k));

  float sz_work;
  magma_int_t info;
  MAGMA_OK(magma_sgeev(MagmaNoVec, MagmaVec, k, nullptr, k, nullptr, nullptr,
                       nullptr, 1, nullptr, k, &sz_work, -1, &info));
  work.resize(sz_work);
}

StreamingDmdGpu::~StreamingDmdGpu() {
  MAGMA_OK(magma_free(dV));
  MAGMA_OK(magma_free(dPhi));
  MAGMA_OK(magma_free(dVSW));
  MAGMA_OK(magma_free(dW));
  MAGMA_OK(magma_free(dA));
  MAGMA_OK(magma_free(dUY));
  MAGMA_OK(magma_free(dVS));
  MAGMA_OK(magma_free(dSigma_inv));
  MAGMA_OK(magma_free(dXY));
}

int StreamingDmdGpu::Run(const float *x, int x_n, bool stream, float *lambda,
                         float *phi, double *elapsed) {
  OK(svd.Run(x, x_n, stream, sigma.data(), v.data(), elapsed));

  real_Double_t time = -magma_sync_wtime(queue);

  magma_scopyvector((n - 1) * (n - 1 - 1), svd.GetDXX() + (n - 1), 1, dXY, 1,
                    queue);
  magma_sgemm(MagmaTrans, MagmaNoTrans, n - 1, 1, m, 1.0f, dX, m,
              dX + m * (n - 1), m, 0.0f, dXY + (n - 1) * (n - 1 - 1), n - 1,
              queue);

  time += magma_sync_wtime(queue);

  auto time_cpu = high_resolution_clock::now();

  invertSigma(sigma, k, 1, sigma_inv);

  time +=
      duration_cast<duration<double>>(high_resolution_clock::now() - time_cpu)
          .count();

  magma_ssetmatrix(n - 1, k, v.data(), n - 1, dV, n - 1, queue);
  magmablas_slaset(MagmaFull, k, k, 0.0f, 0.0f, dSigma_inv, k, queue);
  magma_ssetvector(k, sigma_inv.data(), 1, dSigma_inv, k + 1, queue);

  time -= magma_sync_wtime(queue);

  magma_sgemm(MagmaNoTrans, MagmaNoTrans, n - 1, k, k, 1., dV, n - 1,
              dSigma_inv, k, 0., dVS, n - 1, queue);
  magma_sgemm(MagmaTrans, MagmaNoTrans, k, n - 1, n - 1, 1., dVS, n - 1, dXY,
              n - 1, 0., dUY, k, queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, k, k, n - 1, 1., dUY, k, dVS, n - 1,
              0., dA, k, queue);

  time += magma_sync_wtime(queue);

  magma_sgetmatrix(k, k, dA, k, a.data(), k, queue);

  time_cpu = high_resolution_clock::now();

  magma_int_t info;
  MAGMA_OK(magma_sgeev(MagmaNoVec, MagmaVec, k, a.data(), k, lambda_real.data(),
                       lambda_imag.data(), nullptr, 1, w.data(), k, work.data(),
                       work.size(), &info));

  time +=
      duration_cast<duration<double>>(high_resolution_clock::now() - time_cpu)
          .count();

  mergeLambda(lambda_real, lambda_imag, k, lambda);
  magma_ssetmatrix(k, k, w.data(), k, dW, k, queue);

  time -= magma_sync_wtime(queue);

  magma_sgemm(MagmaNoTrans, MagmaNoTrans, n - 1, k, k, 1.0f, dVS, n - 1, dW, k,
              0.0f, dVSW, n - 1, queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, m, k, n - 1, 1.0f, dX + m, m, dVSW,
              n - 1, 0.0f, dPhi, m, queue);

  time += magma_sync_wtime(queue);

  if (phi)
    magma_sgetmatrix(m, k, dPhi, m, phi, m, queue);

  if (elapsed)
    *elapsed += time;

  return 0;
}