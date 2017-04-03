/*
    Copyright (c) 2016-7 Seth Pendergrass. See LICENSE.
*/
#pragma once
#include "ssvd.h"

namespace ssvd {

class Sdmd {
 public:
  virtual ~Sdmd(){};
  virtual int Run(const float *x, int x_n, bool stream, float *lambda,
                  float *phi, double *elapsed = nullptr, int k_sigma = 0) = 0;
};

class SdmdCpu : public Sdmd {
 public:
  SdmdCpu(int m, int n);
  virtual ~SdmdCpu(){};
  virtual int Run(const float *x, int n, bool stream, float *lambda, float *phi,
                  double *elapsed = nullptr, int k_sigma = 0) override;
  const float *GetSigma() const { return sigma.data(); }
  const float *GetV() const { return v.data(); }
  const float *GetW() const { return w.data(); }

 protected:
  int m, n, k;
  SsvdCpu svd;
  std::vector<float> sigma, v, w, lambda_real, lambda_imag, phi, xy,
      sigma_inv_mat, vs, uy, a, vsw;
};

class SdmdMagma : public Sdmd {
 public:
  SdmdMagma(int m, int n);
  virtual ~SdmdMagma();
  virtual int Run(const float *x, int n, bool stream, float *lambda, float *phi,
                  double *elapsed = nullptr, int k_sigma = 0) override;
  magma_queue_t GetQueue() const { return queue; }
  magmaFloat_ptr GetDSigmaInv() const { return dSigma_inv; }
  magmaFloat_ptr GetDV() const { return dV; }
  magmaFloat_ptr GetDW() const { return dW; }
  magmaFloat_ptr GetDPhi() const { return dPhi; }

 protected:
  int m, n, k;
  SsvdMagma svd;
  magmaFloat_ptr dXY, dSigma_inv, dVS, dUY, dA, dW, dVSW, dPhi, dV, dX;
  std::vector<float> sigma, v, w, a, lambda_real, lambda_imag, sigma_inv;
  magma_queue_t queue;
};
}