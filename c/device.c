// tensor device registry (Phase 1: OpenCL backend).
//
// Part of the Alya Language package ecosystem (https://github.com/alya-lang).
//
// The registry delegates discovery to the portable OpenCL backend in
// `c/ocl.c` (runtime-loaded, no SDK or link dependency). When no OpenCL
// platform exists, the count is 0 and all compute falls back to the Alya
// CPU/SIMD kernels in `src/lib.alya` — loudly documented, never a silent
// fake GPU. Metal / CUDA remain tracked follow-ups (see README roadmap).

#include <stdint.h>

int32_t alya_tensor_ocl_count(void);
int32_t alya_tensor_ocl_supports(int32_t dtype);

int32_t alya_tensor_device_count(void) {
    return alya_tensor_ocl_count();
}

int32_t alya_tensor_device_supports(int32_t dtype) {
    return alya_tensor_ocl_supports(dtype);
}
