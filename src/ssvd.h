/*
    Copyright (c) 2016-7 Seth Pendergrass. See LICENSE.
*/
#pragma once
#include <magma_v2.h>
#include <vector>

namespace ssvd {

class Ssvd {
 public:
  virtual ~Ssvd(){};
  virtual int Run(const float *x, int x_n, bool stream, float *sigma, float *v,
                  double *elapsed = nullptr) = 0;
};

class SsvdCpu : public Ssvd {
 public:
  SsvdCpu(int m, int n, int k);
  virtual ~SsvdCpu(){};
  virtual int Run(const float *x, int n, bool stream, float *sigma, float *v,
                  double *elapsed = nullptr) override;
  const float *GetXX() const { return xx.data(); }

 protected:
  int m, n;
  std::vector<float> xx, v, sigma;
  std::vector<int> isuppz;
};

class SsvdMagma : public Ssvd {
 public:
  SsvdMagma(int m, int n, int k, int n_full = 0);
  virtual ~SsvdMagma();
  virtual int Run(const float *x, int n, bool stream, float *sigma, float *v,
                  double *elapsed = nullptr) override;
  magma_queue_t GetQueue() const { return queue; }
  magmaFloat_ptr GetDX() const { return dX; }
  magmaFloat_ptr GetDXX() const { return dXX; }

 protected:
  int m, n, n_full;
  magma_queue_t queue;
  magmaFloat_ptr dX, dXX, dV;
  std::vector<float> sigma, wA, work;
  std::vector<int> iwork;
};
}