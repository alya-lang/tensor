// tensor device registry (Phase 1: OpenCL + Metal backends).
//
// Part of the Alya Language package ecosystem (https://github.com/alya-lang).
//
// The registry aggregates native backends and routes every entry point to
// exactly one active backend: Metal first on macOS (Apple GPUs have no
// OpenCL path going forward), OpenCL everywhere else. `c/ocl.c` loads the
// system OpenCL library at runtime; `c/device_metal.c` links Metal +
// Foundation on macOS only (compiled via `c-sources-macos`, guarded below
// so other platforms link cleanly). With no backend, every entry reports
// unavailable and the Alya side falls back to CPU/SIMD.

#include <stdint.h>
#include <stddef.h>

int32_t alya_tensor_ocl_count(void);
int32_t alya_tensor_ocl_supports(int32_t dtype);
const char *alya_tensor_ocl_error(void);
const char *alya_tensor_ocl_name(void);
void *alya_tensor_ocl_alloc(int32_t byte_size);
void alya_tensor_ocl_free(void *handle);
int32_t alya_tensor_ocl_write(void *handle, void *host, int32_t byte_size);
int32_t alya_tensor_ocl_read(void *handle, void *host, int32_t byte_size);
int32_t alya_tensor_ocl_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype);
int32_t alya_tensor_ocl_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype);
int32_t alya_tensor_ocl_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off,
                               int32_t r_off, int32_t m, int32_t n, int32_t k, int32_t dtype);
int32_t alya_tensor_ocl_sync(void);

#ifdef __APPLE__
int32_t alya_tensor_metal_count(void);
int32_t alya_tensor_metal_supports(int32_t dtype);
const char *alya_tensor_metal_error(void);
const char *alya_tensor_metal_name(void);
void *alya_tensor_metal_alloc(int32_t byte_size);
void alya_tensor_metal_free(void *handle);
int32_t alya_tensor_metal_write(void *handle, void *host, int32_t byte_size);
int32_t alya_tensor_metal_read(void *handle, void *host, int32_t byte_size);
int32_t alya_tensor_metal_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype);
int32_t alya_tensor_metal_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype);
int32_t alya_tensor_metal_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off,
                                 int32_t r_off, int32_t m, int32_t n, int32_t k, int32_t dtype);
int32_t alya_tensor_metal_sync(void);

static int use_metal_backend(void) {
    static int decided = 0;
    static int metal = 0;
    if (!decided) {
        metal = alya_tensor_metal_count() > 0;
        decided = 1;
    }
    return metal;
}
#endif

int32_t alya_tensor_device_count(void) {
#ifdef __APPLE__
    {
        int32_t mc = alya_tensor_metal_count();
        if (mc > 0) return mc;
    }
#endif
    return alya_tensor_ocl_count();
}

int32_t alya_tensor_device_supports(int32_t dtype) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_supports(dtype);
#endif
    return alya_tensor_ocl_supports(dtype);
}

const char *alya_tensor_device_error(void) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_error();
#endif
    return alya_tensor_ocl_error();
}

const char *alya_tensor_device_name(void) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_name();
#endif
    return alya_tensor_ocl_name();
}

void *alya_tensor_device_alloc(int32_t byte_size) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_alloc(byte_size);
#endif
    return alya_tensor_ocl_alloc(byte_size);
}

void alya_tensor_device_free(void *handle) {
#ifdef __APPLE__
    if (use_metal_backend()) {
        alya_tensor_metal_free(handle);
        return;
    }
#endif
    alya_tensor_ocl_free(handle);
}

int32_t alya_tensor_device_write(void *handle, void *host, int32_t byte_size) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_write(handle, host, byte_size);
#endif
    return alya_tensor_ocl_write(handle, host, byte_size);
}

int32_t alya_tensor_device_read(void *handle, void *host, int32_t byte_size) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_read(handle, host, byte_size);
#endif
    return alya_tensor_ocl_read(handle, host, byte_size);
}

int32_t alya_tensor_device_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_add(ah, bh, oh, count, dtype);
#endif
    return alya_tensor_ocl_add(ah, bh, oh, count, dtype);
}

int32_t alya_tensor_device_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_mul(ah, bh, oh, count, dtype);
#endif
    return alya_tensor_ocl_mul(ah, bh, oh, count, dtype);
}

int32_t alya_tensor_device_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off,
                                  int32_t r_off, int32_t m, int32_t n, int32_t k, int32_t dtype) {
#ifdef __APPLE__
    if (use_metal_backend()) {
        return alya_tensor_metal_matmul(ah, bh, oh, a_off, b_off, r_off, m, n, k, dtype);
    }
#endif
    return alya_tensor_ocl_matmul(ah, bh, oh, a_off, b_off, r_off, m, n, k, dtype);
}

int32_t alya_tensor_device_sync(void) {
#ifdef __APPLE__
    if (use_metal_backend()) return alya_tensor_metal_sync();
#endif
    return alya_tensor_ocl_sync();
}
