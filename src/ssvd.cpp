/*
 * Copyright 2016-20 Seth Pendergrass. See LICENSE.
 */

#include "ssvd.h"
#include "dmd.h"
#include "svd.h"
#include <memory>
using namespace std;
using namespace ssvd;

unique_ptr<StreamingSvd> svd;
unique_ptr<StreamingDmd> dmd;
double last_time;

int Svd(const float *x, int m, int n, int k, bool gpu, bool streaming,
        float *sigma, float *v) {
  if (k == 0)
    k = n;
  assert(k <= n);

  if (gpu)
    svd = make_unique<StreamingSvdGpu>(m, n, k);
  else
    svd = make_unique<StreamingSvdCpu>(m, n, k);

  int res = svd->Run(x, n, false, sigma, v, &last_time);

  if (res || !streaming)
    svd.reset();

  return res;
}

int SvdUpdate(const float *x_new, int n_new, float *sigma, float *v) {
  return svd->Run(x_new, n_new, true, sigma, v, &last_time);
}

void SvdStop() { svd.reset(); }

int Dmd(const float *x, int m, int n, int k, bool gpu, bool streaming,
        float *lambda, float *phi) {
  if (k == 0)
    k = n - 1;
  assert(k < n);

  if (gpu)
    dmd = make_unique<StreamingDmdGpu>(m, n, k);
  else
    dmd = make_unique<StreamingDmdCpu>(m, n, k);

  int res = dmd->Run(x, n, false, lambda, phi, &last_time);

  if (res || !streaming)
    dmd.reset();

  return res;
}

int DmdUpdate(const float *x_new, int n_new, float *lambda, float *phi) {
  return dmd->Run(x_new, n_new, true, lambda, phi, &last_time);
}

void DmdStop() { dmd.reset(); }

double GetElapsedCalcs() { return last_time; }
