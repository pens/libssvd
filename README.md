# libssvd
libssvd is an open-source package providing the Streaming Singular Value Decomposition (SVD) and Dynamic Mode Decomposition (DMD). Both algorithms are implemented for CPU and GPU.

libssvd was created as part of [*Streaming GPU Singular Value and Dynamic Mode Decompositions*](https://arxiv.org/abs/1612.07875).

## Dependencies
- MAGMA (http://icl.cs.utk.edu/magma/software/) \*
- CUDA
- BLAS \*
- Doxygen (if building documentation)

\* Included for Windows

## Style
Style is enforced through clang-format: `clang-format -style=file -i include/* src/*`

## Documentation
To build documentation: `doxygen`
