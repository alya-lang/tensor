// tensor device registry (Phase 0).
//
// Part of the Alya Language package ecosystem (https://github.com/alya-lang).
//
// Accelerator registry for tensor compute offload. No platform kernel
// (Metal / CUDA / OpenCL) is registered yet, so the count is 0 on every
// OS and all compute falls back to the Alya CPU/SIMD kernels in
// `src/lib.alya`. Platform backends register here as they land; the Alya
// side (`src/core/device.alya`) already routes through this registry, so
// user code written against `Tensor.to()` needs no changes.
//
// Contract:
// - `alya_tensor_device_count()` returns the number of usable accelerator
//   devices (0 today: honest fallback, never a silent fake GPU).
// - Future kernels take raw buffer pointers (`void *`) plus element
//   counts, because Alya arrays do not marshal to C pointers; only
//   scalars cross the FFI boundary.

#include <stdint.h>

int32_t alya_tensor_device_count(void) {
    return 0;
}
