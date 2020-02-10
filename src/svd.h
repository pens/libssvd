#pragma once
#include "common.h"
#include <vector>

namespace ssvd {

class StreamingSvd {
public:
  StreamingSvd() = default;
  virtual ~StreamingSvd() = default;
  StreamingSvd(const StreamingSvd &) = delete;
  StreamingSvd &operator=(const StreamingSvd &) = delete;
  StreamingSvd(StreamingSvd &&) = delete;
  StreamingSvd &&operator=(StreamingSvd &&) = delete;

  virtual int Run(const float *x, int x_n, bool stream, float *sigma, float *v,
                  double *elapsed = nullptr) = 0;
};

class StreamingSvdCpu : public StreamingSvd {
public:
  StreamingSvdCpu(int m, int n, int k);
  virtual ~StreamingSvdCpu() = default;
  StreamingSvdCpu(const StreamingSvdCpu &) = delete;
  StreamingSvdCpu &operator=(const StreamingSvdCpu &) = delete;
  StreamingSvdCpu(StreamingSvdCpu &&) = delete;
  StreamingSvdCpu &&operator=(StreamingSvdCpu &&) = delete;

  int Run(const float *x, int n, bool stream, float *sigma, float *v,
          double *elapsed = nullptr) override;

  const float *GetXX() const { return xx.data(); }

private:
  int m;
  int n;
  int k;
  std::vector<float> xx;
  std::vector<float> xx_temp;
  std::vector<float> sigma;
  std::vector<int> isuppz;
  std::vector<float> work;
  std::vector<int> iwork;
};

class StreamingSvdGpu : public StreamingSvd {
public:
  StreamingSvdGpu(int m, int n, int k, int n_full = 0);
  virtual ~StreamingSvdGpu();
  StreamingSvdGpu(const StreamingSvdGpu &) = delete;
  StreamingSvdGpu &operator=(const StreamingSvdGpu &) = delete;
  StreamingSvdGpu(StreamingSvdGpu &&) = delete;
  StreamingSvdGpu &&operator=(StreamingSvdGpu &&) = delete;

  int Run(const float *x, int n, bool stream, float *sigma, float *v,
          double *elapsed = nullptr) override;

  magma_queue_t GetQueue() const { return queue; }
  magmaFloat_const_ptr GetDX() const { return dX; }
  magmaFloat_const_ptr GetDXX() const { return dXX; }

private:
  int m;
  int n;
  int k;
  int n_full;
  magma_queue_t queue;
  magmaFloat_ptr dX;
  magmaFloat_ptr dXX;
  magmaFloat_ptr dXX_dV;
  std::vector<float> sigma_temp;
  std::vector<float> wA;
  std::vector<float> work;
  std::vector<int> iwork;
};
} // namespace ssvd