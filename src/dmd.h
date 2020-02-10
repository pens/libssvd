#pragma once
#include "svd.h"

namespace ssvd {

class StreamingDmd {
public:
  StreamingDmd() = default;
  virtual ~StreamingDmd() = default;
  StreamingDmd(const StreamingDmd &) = delete;
  StreamingDmd &operator=(const StreamingDmd &) = delete;
  StreamingDmd(StreamingDmd &&) = delete;
  StreamingDmd &&operator=(StreamingDmd &&) = delete;

  virtual int Run(const float *x, int x_n, bool stream, float *lambda,
                  float *phi, double *elapsed) = 0;
};

class StreamingDmdCpu : public StreamingDmd {
public:
  StreamingDmdCpu(int m, int n, int k);
  virtual ~StreamingDmdCpu() = default;
  StreamingDmdCpu(const StreamingDmdCpu &) = delete;
  StreamingDmdCpu &operator=(const StreamingDmdCpu &) = delete;
  StreamingDmdCpu(StreamingDmdCpu &&) = delete;
  StreamingDmdCpu &&operator=(StreamingDmdCpu &&) = delete;

  int Run(const float *x, int n, bool stream, float *lambda, float *phi,
          double *elapsed) override;

  const float *GetSigma() const { return sigma.data(); }
  const float *GetV() const { return v.data(); }
  const float *GetW() const { return w.data(); }

private:
  int m;
  int n;
  int k;
  StreamingSvdCpu svd;
  std::vector<float> sigma;
  std::vector<float> v;
  std::vector<float> w;
  std::vector<float> lambda_real;
  std::vector<float> lambda_imag;
  std::vector<float> phi;
  std::vector<float> xy;
  std::vector<float> sigma_inv_mat;
  std::vector<float> vs;
  std::vector<float> uy;
  std::vector<float> a;
  std::vector<float> vsw;
  std::vector<float> work;
};

class StreamingDmdGpu : public StreamingDmd {
public:
  StreamingDmdGpu(int m, int n, int k);
  virtual ~StreamingDmdGpu();
  StreamingDmdGpu(const StreamingDmdGpu &) = delete;
  StreamingDmdGpu &operator=(const StreamingDmdGpu &) = delete;
  StreamingDmdGpu(StreamingDmdGpu &&) = delete;
  StreamingDmdGpu &&operator=(StreamingDmdGpu &&) = delete;

  int Run(const float *x, int n, bool stream, float *lambda, float *phi,
          double *elapsed) override;

  magma_queue_t GetQueue() const { return queue; }
  magmaFloat_const_ptr GetDSigmaInv() const { return dSigma_inv; }
  magmaFloat_const_ptr GetDV() const { return dV; }
  magmaFloat_const_ptr GetDW() const { return dW; }
  magmaFloat_const_ptr GetDPhi() const { return dPhi; }

private:
  int m;
  int n;
  int k;
  StreamingSvdGpu svd;
  magmaFloat_ptr dXY;
  magmaFloat_ptr dSigma_inv;
  magmaFloat_ptr dVS;
  magmaFloat_ptr dUY;
  magmaFloat_ptr dA;
  magmaFloat_ptr dW;
  magmaFloat_ptr dVSW;
  magmaFloat_ptr dPhi;
  magmaFloat_ptr dV;
  magmaFloat_const_ptr dX;
  std::vector<float> sigma;
  std::vector<float> v;
  std::vector<float> w;
  std::vector<float> a;
  std::vector<float> lambda_real;
  std::vector<float> lambda_imag;
  std::vector<float> sigma_inv;
  std::vector<float> work;
  magma_queue_t queue;
};
} // namespace ssvd
