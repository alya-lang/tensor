// tensor Metal compute backend (macOS only, pure C over ObjC runtime).
//
// Part of the Alya Language package ecosystem (https://github.com/alya-lang).
//
// Same contract as c/ocl.c, driven through the Objective-C runtime C API
// with no ObjC syntax and no SDK headers (gui/c/cocoa_window.c precedent),
// so `gcc -fsyntax-only` passes anywhere while linking needs macOS
// frameworks (manifest `c-link-flags-macos`). MSL kernels compile at
// runtime via newLibraryWithSource; Apple GPUs have no fp64, so only
// f32/i32/i64 are covered (f64 falls back to CPU). All dispatch is
// synchronous (waitUntilCompleted); the async fence is trivially
// satisfied. Compiled into the package on macOS only.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// --- Objective-C runtime C API (declared manually, no headers needed) ---

typedef struct objc_class *Class;
typedef struct objc_object *id;
typedef const struct objc_selector *SEL;
typedef unsigned long NSUInteger;

typedef struct MTLSize {
    NSUInteger width;
    NSUInteger height;
    NSUInteger depth;
} MTLSize;

extern Class objc_getClass(const char *name);
extern SEL sel_registerName(const char *name);
extern id objc_msgSend(id self, SEL op, ...);
extern id MTLCreateSystemDefaultDevice(void);

// MTLCommandBufferStatusCompleted.
#define MTL_STATUS_COMPLETED 4

// --- MSL kernels (runtime-compiled; f32/i32/i64 only, no fp64 on Apple GPUs) ---

static const char *metal_kernel_src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void tadd_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],\n"
    "                     device float* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    o[i] = a[i] + b[i];\n"
    "}\n"
    "kernel void tmul_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],\n"
    "                     device float* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    o[i] = a[i] * b[i];\n"
    "}\n"
    "kernel void tadd_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]],\n"
    "                     device int* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    o[i] = a[i] + b[i];\n"
    "}\n"
    "kernel void tmul_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]],\n"
    "                     device int* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    o[i] = a[i] * b[i];\n"
    "}\n"
    "kernel void tadd_i64(device const long* a [[buffer(0)]], device const long* b [[buffer(1)]],\n"
    "                     device long* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    o[i] = a[i] + b[i];\n"
    "}\n"
    "kernel void tmul_i64(device const long* a [[buffer(0)]], device const long* b [[buffer(1)]],\n"
    "                     device long* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "    o[i] = a[i] * b[i];\n"
    "}\n"
    "kernel void tmm_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],\n"
    "                    device float* C [[buffer(2)]], constant int* P [[buffer(3)]],\n"
    "                    uint2 gid [[thread_position_in_grid]]) {\n"
    "    int j = (int)gid.x;\n"
    "    int i = (int)gid.y;\n"
    "    int ao = P[0]; int bo = P[1]; int ro = P[2];\n"
    "    int M = P[3]; int N = P[4]; int K = P[5];\n"
    "    float s = 0.0f;\n"
    "    for (int p = 0; p < K; ++p) s += A[ao + i * K + p] * B[bo + p * N + j];\n"
    "    C[ro + i * N + j] = s;\n"
    "}\n"
    "kernel void tmm_i32(device const int* A [[buffer(0)]], device const int* B [[buffer(1)]],\n"
    "                    device int* C [[buffer(2)]], constant int* P [[buffer(3)]],\n"
    "                    uint2 gid [[thread_position_in_grid]]) {\n"
    "    int j = (int)gid.x;\n"
    "    int i = (int)gid.y;\n"
    "    int ao = P[0]; int bo = P[1]; int ro = P[2];\n"
    "    int M = P[3]; int N = P[4]; int K = P[5];\n"
    "    int s = 0;\n"
    "    for (int p = 0; p < K; ++p) s += A[ao + i * K + p] * B[bo + p * N + j];\n"
    "    C[ro + i * N + j] = s;\n"
    "}\n"
    "kernel void tmm_i64(device const long* A [[buffer(0)]], device const long* B [[buffer(1)]],\n"
    "                    device long* C [[buffer(2)]], constant int* P [[buffer(3)]],\n"
    "                    uint2 gid [[thread_position_in_grid]]) {\n"
    "    int j = (int)gid.x;\n"
    "    int i = (int)gid.y;\n"
    "    int ao = P[0]; int bo = P[1]; int ro = P[2];\n"
    "    int M = P[3]; int N = P[4]; int K = P[5];\n"
    "    long s = 0;\n"
    "    for (int p = 0; p < K; ++p) s += A[ao + i * K + p] * B[bo + p * N + j];\n"
    "    C[ro + i * N + j] = s;\n"
    "}\n";

static const char *metal_kernel_names[9] = {
    "tadd_f32", "tmul_f32",
    "tadd_i32", "tmul_i32",
    "tadd_i64", "tmul_i64",
    "tmm_f32", "tmm_i32", "tmm_i64"
};

// --- Backend state (single device, process lifetime) ---

static int metal_state = 0; // 0 = unprobed, 1 = ready, -1 = unavailable
static id metal_dev = 0;
static id metal_queue = 0;
static id metal_pipes[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static char metal_error[2048] = {0};
static char metal_name[256] = {0};

static void metal_fail(const char *msg) {
    size_t n = 0;
    while (n + 1 < sizeof(metal_error) && msg[n]) {
        metal_error[n] = msg[n];
        ++n;
    }
    metal_error[n] = 0;
}

static void metal_trace(const char *what) {
    const char *on = getenv("ALYA_TENSOR_OCL_DEBUG");
    if (on && on[0]) {
        fprintf(stderr, "[tensor-metal] %s\n", what);
    }
}

static id metal_nsstring(const char *cstr) {
    Class cls = objc_getClass("NSString");
    SEL s_alloc = sel_registerName("alloc");
    SEL s_init = sel_registerName("initWithUTF8String:");
    id s = ((id(*)(id, SEL))objc_msgSend)((id)cls, s_alloc);
    if (!s) return 0;
    return ((id(*)(id, SEL, const char *))objc_msgSend)(s, s_init, cstr);
}

static void metal_release(id obj) {
    SEL s = sel_registerName("release");
    if (obj) ((void(*)(id, SEL))objc_msgSend)(obj, s);
}

static void metal_describe_error(id err) {
    SEL s_desc = sel_registerName("localizedDescription");
    SEL s_utf8 = sel_registerName("UTF8String");
    id text = 0;
    const char *c = 0;
    size_t n = 0;
    if (!err) {
        metal_fail("Metal operation failed");
        return;
    }
    text = ((id(*)(id, SEL))objc_msgSend)(err, s_desc);
    if (!text) {
        metal_fail("Metal operation failed");
        return;
    }
    c = ((const char *(*)(id, SEL))objc_msgSend)(text, s_utf8);
    if (!c) {
        metal_fail("Metal operation failed");
        return;
    }
    while (n + 1 < sizeof(metal_error) && c[n]) {
        metal_error[n] = c[n];
        ++n;
    }
    metal_error[n] = 0;
}

static void metal_init(void) {
    SEL s_lib = sel_registerName("newLibraryWithSource:options:error:");
    SEL s_fn = sel_registerName("newFunctionWithName:");
    SEL s_pipe = sel_registerName("newComputePipelineStateWithFunction:error:");
    SEL s_queue = sel_registerName("newCommandQueue");
    SEL s_name = sel_registerName("name");
    SEL s_utf8 = sel_registerName("UTF8String");
    id src = 0;
    id lib = 0;
    id err = 0;
    int ki = 0;

    if (metal_state != 0) return;
    metal_state = -1;
    metal_dev = MTLCreateSystemDefaultDevice();
    if (!metal_dev) {
        metal_fail("no Metal system device");
        return;
    }
    src = metal_nsstring(metal_kernel_src);
    if (!src) {
        metal_fail("NSString allocation failed");
        return;
    }
    lib = ((id(*)(id, SEL, id, id, id *))objc_msgSend)(metal_dev, s_lib, src, 0, &err);
    metal_release(src);
    if (!lib) {
        metal_describe_error(err);
        return;
    }
    for (ki = 0; ki < 9; ++ki) {
        id fname = 0;
        id fn = 0;
        id pipe = 0;
        id perr = 0;
        fname = metal_nsstring(metal_kernel_names[ki]);
        if (!fname) {
            metal_fail("NSString allocation failed");
            metal_release(lib);
            return;
        }
        fn = ((id(*)(id, SEL, id))objc_msgSend)(lib, s_fn, fname);
        metal_release(fname);
        if (!fn) {
            metal_fail("newFunctionWithName failed");
            metal_release(lib);
            return;
        }
        pipe = ((id(*)(id, SEL, id, id *))objc_msgSend)(metal_dev, s_pipe, fn, &perr);
        metal_release(fn);
        if (!pipe) {
            metal_describe_error(perr);
            metal_release(lib);
            return;
        }
        metal_pipes[ki] = pipe;
    }
    metal_release(lib);
    metal_queue = ((id(*)(id, SEL))objc_msgSend)(metal_dev, s_queue);
    if (!metal_queue) {
        metal_fail("newCommandQueue failed");
        return;
    }
    {
        id dname = ((id(*)(id, SEL))objc_msgSend)(metal_dev, s_name);
        const char *c = 0;
        size_t n = 0;
        if (dname) {
            c = ((const char *(*)(id, SEL))objc_msgSend)(dname, s_utf8);
            while (c && n + 1 < sizeof(metal_name) && c[n]) {
                metal_name[n] = c[n];
                ++n;
            }
        }
        metal_name[n] = 0;
    }
    metal_state = 1;
    metal_trace("backend ready");
}

// Kernel table index by (op, dtype): op 0 = add, 1 = mul, 2 = matmul;
// dtype 1 = f32, 2 = i32, 3 = i64 (f64 unsupported: index -1).
static int metal_kernel_index(int op, int dtype) {
    static const int table[3][4] = {
        {-1, 0, 2, 4},
        {-1, 1, 3, 5},
        {-1, 6, 7, 8}
    };
    if (op < 0 || op > 2 || dtype < 0 || dtype > 3) return -1;
    return table[op][dtype];
}

struct metal_buffer {
    id buf;
};

static id metal_cmdbuf(void) {
    SEL s = sel_registerName("commandBuffer");
    return ((id(*)(id, SEL))objc_msgSend)(metal_queue, s);
}

static int metal_run(id pipe, struct metal_buffer *ah, struct metal_buffer *bh, struct metal_buffer *oh,
                     const int *params, int param_count, unsigned long gx, unsigned long gy) {
    SEL s_enc = sel_registerName("computeCommandEncoder");
    SEL s_state = sel_registerName("setComputePipelineState:");
    SEL s_buf = sel_registerName("setBuffer:offset:atIndex:");
    SEL s_bytes = sel_registerName("setBytes:length:atIndex:");
    SEL s_disp = sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:");
    SEL s_end = sel_registerName("endEncoding");
    SEL s_commit = sel_registerName("commit");
    SEL s_wait = sel_registerName("waitUntilCompleted");
    SEL s_status = sel_registerName("status");
    id cmd = 0;
    id enc = 0;
    long st = 0;
    MTLSize grid;
    MTLSize tgs;
    grid.width = (NSUInteger)gx;
    grid.height = (NSUInteger)gy;
    grid.depth = 1;
    if (gy <= 1) {
        tgs.width = 256;
        tgs.height = 1;
        tgs.depth = 1;
    } else {
        tgs.width = 16;
        tgs.height = 16;
        tgs.depth = 1;
    }

    cmd = metal_cmdbuf();
    if (!cmd) {
        metal_fail("commandBuffer failed");
        return 0;
    }
    enc = ((id(*)(id, SEL))objc_msgSend)(cmd, s_enc);
    if (!enc) {
        metal_fail("computeCommandEncoder failed");
        metal_release(cmd);
        return 0;
    }
    ((void(*)(id, SEL, id))objc_msgSend)(enc, s_state, pipe);
    ((void(*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(enc, s_buf, ah->buf, (NSUInteger)0, (NSUInteger)0);
    ((void(*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(enc, s_buf, bh->buf, (NSUInteger)0, (NSUInteger)1);
    ((void(*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(enc, s_buf, oh->buf, (NSUInteger)0, (NSUInteger)2);
    if (params && param_count > 0) {
        ((void(*)(id, SEL, const void *, NSUInteger, NSUInteger))objc_msgSend)(
            enc, s_bytes, (const void *)params, (NSUInteger)(param_count * (int)sizeof(int)), (NSUInteger)3);
    }
    ((void(*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(enc, s_disp, grid, tgs);
    ((void(*)(id, SEL))objc_msgSend)(enc, s_end);
    metal_release(enc);
    ((void(*)(id, SEL))objc_msgSend)(cmd, s_commit);
    ((void(*)(id, SEL))objc_msgSend)(cmd, s_wait);
    st = ((long(*)(id, SEL))objc_msgSend)(cmd, s_status);
    metal_release(cmd);
    if (st != MTL_STATUS_COMPLETED) {
        metal_fail("command buffer did not complete");
        return 0;
    }
    return 1;
}

// --- Public entry points (called from device.c) ---

int32_t alya_tensor_metal_count(void) {
    metal_init();
    if (metal_state != 1) return 0;
    return 1;
}

int32_t alya_tensor_metal_supports(int32_t dtype) {
    metal_init();
    if (metal_state != 1) return 0;
    if (dtype == 1 || dtype == 2 || dtype == 3) return 1;
    return 0;
}

const char *alya_tensor_metal_error(void) {
    return metal_error;
}

const char *alya_tensor_metal_name(void) {
    metal_init();
    return metal_name;
}

void *alya_tensor_metal_alloc(int32_t byte_size) {
    SEL s = sel_registerName("newBufferWithLength:options:");
    struct metal_buffer *h = 0;
    id buf = 0;
    metal_init();
    if (metal_state != 1 || byte_size <= 0) return 0;
    buf = ((id(*)(id, SEL, NSUInteger, NSUInteger))objc_msgSend)(
        metal_dev, s, (NSUInteger)byte_size, (NSUInteger)0);
    if (!buf) return 0;
    h = (struct metal_buffer *)malloc(sizeof(struct metal_buffer));
    if (!h) {
        metal_release(buf);
        return 0;
    }
    h->buf = buf;
    return (void *)h;
}

void alya_tensor_metal_free(void *handle) {
    struct metal_buffer *h = (struct metal_buffer *)handle;
    if (h) {
        metal_release(h->buf);
        free(h);
    }
}

static int metal_copy(struct metal_buffer *h, void *host, int32_t byte_size, int to_device) {
    SEL s = sel_registerName("contents");
    void *p = 0;
    size_t n = 0;
    if (!h || !h->buf || !host || byte_size <= 0) return 0;
    p = ((void *(*)(id, SEL))objc_msgSend)(h->buf, s);
    if (!p) return 0;
    n = (size_t)byte_size;
    if (to_device) {
        memcpy(p, host, n);
    } else {
        memcpy(host, p, n);
    }
    return 1;
}

int32_t alya_tensor_metal_write(void *handle, void *host, int32_t byte_size) {
    metal_init();
    if (metal_state != 1) return 0;
    return metal_copy((struct metal_buffer *)handle, host, byte_size, 1);
}

int32_t alya_tensor_metal_read(void *handle, void *host, int32_t byte_size) {
    metal_init();
    if (metal_state != 1) return 0;
    return metal_copy((struct metal_buffer *)handle, host, byte_size, 0);
}

static int32_t metal_launch_ew(void *ah, void *bh, void *oh, int32_t count, int op, int32_t dtype) {
    int ki = 0;
    metal_init();
    if (metal_state != 1 || !ah || !bh || !oh || count <= 0) return 0;
    ki = metal_kernel_index(op, (int)dtype);
    if (ki < 0 || !metal_pipes[ki]) return 0;
    if (!metal_run(metal_pipes[ki], (struct metal_buffer *)ah, (struct metal_buffer *)bh,
                   (struct metal_buffer *)oh, 0, 0, (unsigned long)count, 1)) {
        return 0;
    }
    metal_trace("element-wise launch ok");
    return 1;
}

int32_t alya_tensor_metal_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    return metal_launch_ew(ah, bh, oh, count, 0, dtype);
}

int32_t alya_tensor_metal_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    return metal_launch_ew(ah, bh, oh, count, 1, dtype);
}

int32_t alya_tensor_metal_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off, int32_t r_off,
                                 int32_t m, int32_t n, int32_t k, int32_t dtype) {
    int params[6];
    int ki = 0;
    metal_init();
    if (metal_state != 1 || !ah || !bh || !oh || m <= 0 || n <= 0 || k <= 0) return 0;
    ki = metal_kernel_index(2, (int)dtype);
    if (ki < 0 || !metal_pipes[ki]) return 0;
    params[0] = a_off;
    params[1] = b_off;
    params[2] = r_off;
    params[3] = m;
    params[4] = n;
    params[5] = k;
    if (!metal_run(metal_pipes[ki], (struct metal_buffer *)ah, (struct metal_buffer *)bh,
                   (struct metal_buffer *)oh, params, 6, (unsigned long)n, (unsigned long)m)) {
        return 0;
    }
    metal_trace("matmul launch ok");
    return 1;
}

int32_t alya_tensor_metal_sync(void) {
    metal_init();
    if (metal_state != 1) return 0;
    return 1;
}
