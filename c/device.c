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
//
// Safety: setting ALYA_TENSOR_NO_GPU=1 disables all probing (pure-CPU
// mode for bisecting hangs). A backend that reports a launch/transfer
// error is poisoned for the process lifetime (fail-safe fallback instead
// of retrying a dead context, which can block forever).

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

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

// --- Staging buffer pool (backend-agnostic free lists) ---
//
// Element-wise and GEMM offload allocate 3 staging buffers per call;
// interactive/bench loops repeat identical sizes, so exact-size matches
// are served from the pool instead of the driver. Entries above 64MB
// bypass the pool. Handles stay valid across pooled reuse because device
// contexts live for the process and the active backend never changes
// after the first decision.
#define DEV_POOL_MAX 16
#define DEV_POOL_MAX_BYTES ((size_t)64 * 1024 * 1024)
#define DEV_LIVE_MAX 256

struct dev_pool_slot {
    int backend;
    void *handle;
    size_t size;
    int used;
};

struct dev_live_entry {
    void *handle;
    size_t size;
    int backend;
    int used;
};

static struct dev_pool_slot dev_pool[DEV_POOL_MAX] = {{0, 0, 0, 0}};
static struct dev_live_entry dev_live[DEV_LIVE_MAX] = {{0, 0, 0, 0}};

static void *pool_take(int back, size_t size) {
    int i = 0;
    int j = 0;
    for (i = 0; i < DEV_POOL_MAX; ++i) {
        if (dev_pool[i].used && dev_pool[i].backend == back && dev_pool[i].size == size) {
            void *h = dev_pool[i].handle;
            dev_pool[i].used = 0;
            dev_pool[i].handle = 0;
            for (j = 0; j < DEV_LIVE_MAX; ++j) {
                if (!dev_live[j].used) {
                    dev_live[j].handle = h;
                    dev_live[j].size = size;
                    dev_live[j].backend = back;
                    dev_live[j].used = 1;
                    break;
                }
            }
            return h;
        }
    }
    return 0;
}

static void live_add(void *h, size_t size, int back) {
    int j = 0;
    if (!h) return;
    for (j = 0; j < DEV_LIVE_MAX; ++j) {
        if (!dev_live[j].used) {
            dev_live[j].handle = h;
            dev_live[j].size = size;
            dev_live[j].backend = back;
            dev_live[j].used = 1;
            return;
        }
    }
}

static int live_remove(void *h, size_t *size, int *back) {
    int j = 0;
    if (!h) return 0;
    for (j = 0; j < DEV_LIVE_MAX; ++j) {
        if (dev_live[j].used && dev_live[j].handle == h) {
            *size = dev_live[j].size;
            *back = dev_live[j].backend;
            dev_live[j].used = 0;
            dev_live[j].handle = 0;
            return 1;
        }
    }
    return 0;
}

static void backend_free(int back, void *h) {
    if (!h) return;
    if (back == 1) {
        alya_tensor_cuda_free(h);
        return;
    }
#ifdef __APPLE__
    if (back == 2) {
        alya_tensor_metal_free(h);
        return;
    }
#endif
    alya_tensor_ocl_free(h);
}

static void *backend_alloc(int back, size_t size) {
    if (back == 1) return alya_tensor_cuda_alloc((int32_t)size);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_alloc((int32_t)size);
#endif
    return alya_tensor_ocl_alloc((int32_t)size);
}

// Active backend: 1 = CUDA, 2 = Metal, 3 = OpenCL, 0 = none. Decided once.
static int active_backend(void) {
    static int decided = 0;
    static int backend = 0;
    if (!decided) {
        const char *off = getenv("ALYA_TENSOR_NO_GPU");
        backend = 0;
        if (!off || !off[0]) {
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
    if (back == 0) return 0;
    if (back == 1) return alya_tensor_cuda_supports(dtype);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_supports(dtype);
#endif
    return alya_tensor_ocl_supports(dtype);
}

static const char *dev_disabled_error = "GPU disabled via ALYA_TENSOR_NO_GPU";
static const char *dev_empty_name = "";

const char *alya_tensor_device_error(void) {
    int back = active_backend();
    if (back == 0) return dev_disabled_error;
    if (back == 1) return alya_tensor_cuda_error();
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_error();
#endif
    return alya_tensor_ocl_error();
}

const char *alya_tensor_device_name(void) {
    int back = active_backend();
    if (back == 0) return dev_empty_name;
    if (back == 1) return alya_tensor_cuda_name();
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_name();
#endif
    return alya_tensor_ocl_name();
}

void *alya_tensor_device_alloc(int32_t byte_size) {
    int back = active_backend();
    size_t size = 0;
    void *h = 0;
    if (back == 0) return 0;
    if (byte_size <= 0) return 0;
    size = (size_t)byte_size;
    h = pool_take(back, size);
    if (h) return h;
    h = backend_alloc(back, size);
    live_add(h, size, back);
    return h;
}

void alya_tensor_device_free(void *handle) {
    size_t size = 0;
    int back = 0;
    int i = 0;
    if (!handle) return;
    if (live_remove(handle, &size, &back)) {
        if (size <= DEV_POOL_MAX_BYTES) {
            for (i = 0; i < DEV_POOL_MAX; ++i) {
                if (!dev_pool[i].used) {
                    dev_pool[i].backend = back;
                    dev_pool[i].handle = handle;
                    dev_pool[i].size = size;
                    dev_pool[i].used = 1;
                    return;
                }
            }
        }
        backend_free(back, handle);
        return;
    }
    {
        int cur = active_backend();
        if (cur != 0) backend_free(cur, handle);
    }
}

int32_t alya_tensor_device_write(void *handle, void *host, int32_t byte_size) {
    int back = active_backend();
    if (back == 0) return 0;
    if (back == 1) return alya_tensor_cuda_write(handle, host, byte_size);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_write(handle, host, byte_size);
#endif
    return alya_tensor_ocl_write(handle, host, byte_size);
}

int32_t alya_tensor_device_read(void *handle, void *host, int32_t byte_size) {
    int back = active_backend();
    if (back == 0) return 0;
    if (back == 1) return alya_tensor_cuda_read(handle, host, byte_size);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_read(handle, host, byte_size);
#endif
    return alya_tensor_ocl_read(handle, host, byte_size);
}

int32_t alya_tensor_device_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    int back = active_backend();
    if (back == 0) return 0;
    if (back == 1) return alya_tensor_cuda_add(ah, bh, oh, count, dtype);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_add(ah, bh, oh, count, dtype);
#endif
    return alya_tensor_ocl_add(ah, bh, oh, count, dtype);
}

int32_t alya_tensor_device_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    int back = active_backend();
    if (back == 0) return 0;
    if (back == 1) return alya_tensor_cuda_mul(ah, bh, oh, count, dtype);
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_mul(ah, bh, oh, count, dtype);
#endif
    return alya_tensor_ocl_mul(ah, bh, oh, count, dtype);
}

int32_t alya_tensor_device_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off,
                                  int32_t r_off, int32_t m, int32_t n, int32_t k, int32_t dtype) {
    int back = active_backend();
    if (back == 0) return 0;
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
    if (back == 0) return 0;
    if (back == 1) return alya_tensor_cuda_sync();
#ifdef __APPLE__
    if (back == 2) return alya_tensor_metal_sync();
#endif
    return alya_tensor_ocl_sync();
}
