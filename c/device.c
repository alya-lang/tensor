// tensor device registry (Phase 1: CUDA + Metal + OpenCL backends).
//
// Part of the Alya Language package ecosystem (https://github.com/alya-lang).
//
// The registry aggregates native backends and routes every entry point to
// exactly one active backend, probed in priority order: CUDA first (any
// OS with an NVIDIA driver; PTX is JIT-compiled, no toolkit needed),
// Metal on macOS (pure C over the ObjC runtime), OpenCL everywhere else
// (runtime-loaded system library). `c/device_cuda.c` and `c/ocl.c` are
// portable (dynamic loading, no link flags); `c/device_metal.c` compiles
// on macOS only (guarded below so other platforms link cleanly). With no
// backend, every entry reports unavailable and the Alya side falls back
// to CPU/SIMD.

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

int32_t alya_tensor_cuda_count(void);
int32_t alya_tensor_cuda_supports(int32_t dtype);
const char *alya_tensor_cuda_error(void);
const char *alya_tensor_cuda_name(void);
void *alya_tensor_cuda_alloc(int32_t byte_size);
void alya_tensor_cuda_free(void *handle);
int32_t alya_tensor_cuda_write(void *handle, void *host, int32_t byte_size);
int32_t alya_tensor_cuda_read(void *handle, void *host, int32_t byte_size);
int32_t alya_tensor_cuda_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype);
int32_t alya_tensor_cuda_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype);
int32_t alya_tensor_cuda_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off,
                                int32_t r_off, int32_t m, int32_t n, int32_t k, int32_t dtype);
int32_t alya_tensor_cuda_sync(void);

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
#endif

// Active backend: 1 = CUDA, 2 = Metal, 3 = OpenCL, 0 = none. Decided once.
static int active_backend(void) {
    static int decided = 0;
    static int backend = 0;
    if (!decided) {
        backend = 0;
        if (alya_tensor_cuda_count() > 0) {
            backend = 1;
        }
#ifdef __APPLE__
        else if (alya_tensor_metal_count() > 0) {
            backend = 2;
        }
#endif
        else if (alya_tensor_ocl_count() > 0) {
            backend = 3;
        }
        decided = 1;
    }
    return backend;
}

int32_t alya_tensor_device_count(void) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_count();
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_count();
#endif
    if (back == 3) return alya_tensor_ocl_count();
    return 0;
}

int32_t alya_tensor_device_supports(int32_t dtype) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_supports(dtype);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_supports(dtype);
#endif
    return alya_tensor_ocl_supports(dtype);
}

const char *alya_tensor_device_error(void) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_error();
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_error();
#endif
    return alya_tensor_ocl_error();
}

const char *alya_tensor_device_name(void) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_name();
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_name();
#endif
    return alya_tensor_ocl_name();
}

void *alya_tensor_device_alloc(int32_t byte_size) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_alloc(byte_size);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_alloc(byte_size);
#endif
    return alya_tensor_ocl_alloc(byte_size);
}

void alya_tensor_device_free(void *handle) {
    int back = active_backend();
    if (back == 1) {
        alya_tensor_cuda_free(handle);
        return;
    }
#ifdef __APPLE__
    if (back == 2) {
        alya_tensor_metal_free(handle);
        return;
    }
#endif
    alya_tensor_ocl_free(handle);
}

int32_t alya_tensor_device_write(void *handle, void *host, int32_t byte_size) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_write(handle, host, byte_size);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_write(handle, host, byte_size);
#endif
    return alya_tensor_ocl_write(handle, host, byte_size);
}

int32_t alya_tensor_device_read(void *handle, void *host, int32_t byte_size) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_read(handle, host, byte_size);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_read(handle, host, byte_size);
#endif
    return alya_tensor_ocl_read(handle, host, byte_size);
}

int32_t alya_tensor_device_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_add(ah, bh, oh, count, dtype);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_add(ah, bh, oh, count, dtype);
#endif
    return alya_tensor_ocl_add(ah, bh, oh, count, dtype);
}

int32_t alya_tensor_device_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_mul(ah, bh, oh, count, dtype);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_mul(ah, bh, oh, count, dtype);
#endif
    return alya_tensor_ocl_mul(ah, bh, oh, count, dtype);
}

int32_t alya_tensor_device_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off,
                                  int32_t r_off, int32_t m, int32_t n, int32_t k, int32_t dtype) {
    int back = active_backend();
    if (back == 1) {
        return alya_tensor_cuda_matmul(ah, bh, oh, a_off, b_off, r_off, m, n, k, dtype);
    }
#ifdef __APPLE__
    if (back == 2) {
        return alya_tensor_metal_matmul(ah, bh, oh, a_off, b_off, r_off, m, n, k, dtype);
    }
#endif
    return alya_tensor_ocl_matmul(ah, bh, oh, a_off, b_off, r_off, m, n, k, dtype);
}

int32_t alya_tensor_device_sync(void) {
    int back = active_backend();
    if (back == 1) return alya_tensor_cuda_sync();
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_sync();
#endif
    return alya_tensor_ocl_sync();
}
