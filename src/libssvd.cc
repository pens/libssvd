/*
    Copyright (c) 2016-7 Seth Pendergrass. See LICENSE.
*/
#include "libssvd.h"
#include <memory>
#include "sdmd.h"
#include "ssvd.h"
using namespace std;
using namespace ssvd;

namespace {
unique_ptr<Ssvd> svd;
unique_ptr<Sdmd> dmd;
}
double last_time;

int SVD(const float *x, int m, int n, int k, bool gpu, bool streaming, float *sigma,
        float *v) {
  if (gpu)
    svd = make_unique<SsvdMagma>(m, n, k);
  else
    svd = make_unique<SsvdCpu>(m, n, k);

  int res = svd->Run(x, n, false, sigma, v, &last_time);

  if (res || !streaming) svd.release();

  return res;
}

int SVDUpdate(const float *x_new, int n_new, float *sigma, float *v) {
  return svd->Run(x_new, n_new, true, sigma, v, &last_time);
}

void SVDStop() { svd.release(); }

int DMD(const float *x, int m, int n, int k, bool gpu, bool streaming, float *lambda,
        float *phi) {
  if (gpu)
    dmd = make_unique<SdmdMagma>(m, n, k);
  else
    dmd = make_unique<SdmdCpu>(m, n, k);

  int res = dmd->Run(x, n, false, lambda, phi, &last_time);

  if (res || !streaming) dmd.release();

  return res;
}

int DMDUpdate(const float *x_new, int n_new, float *lambda, float *phi) {
  return dmd->Run(x_new, n_new, true, lambda, phi, &last_time);
}

void DMDStop() { dmd.release(); }

void GetElapsed(double *time) { *time = last_time; }
