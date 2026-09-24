// tensor OpenCL compute backend (portable, dynamically loaded).
//
// Part of the Alya Language package ecosystem (https://github.com/alya-lang).
//
// No OpenCL SDK headers are required: the 1.2 API subset is loaded at
// runtime via LoadLibrary (Windows) or dlopen (Linux/macOS) with
// hand-declared prototypes, so package builds never gain a link
// dependency. When no OpenCL platform exists, every entry point reports
// unavailable (0/NULL) and the Alya side falls back to CPU/SIMD.
//
// Covered dtypes: f32 + i32 + i64 are OpenCL C core; f64 needs the
// `cl_khr_fp64` device extension and is compiled in only when present
// (queried via CL_DEVICE_DOUBLE_FP_CONFIG).

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
static HMODULE ocl_lib = NULL;
static void *ocl_sym(const char *name) {
    return (void *)GetProcAddress(ocl_lib, name);
}
static int ocl_lib_open(void) {
    ocl_lib = LoadLibraryA("OpenCL.dll");
    return ocl_lib != NULL;
}
#else
#include <dlfcn.h>
static void *ocl_lib = NULL;
static void *ocl_sym(const char *name) {
    return dlsym(ocl_lib, name);
}
static int ocl_lib_open(void) {
#if defined(__APPLE__)
    ocl_lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW);
    if (!ocl_lib) {
        ocl_lib = dlopen("/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL", RTLD_NOW);
    }
#else
    ocl_lib = dlopen("libOpenCL.so.1", RTLD_NOW);
    if (!ocl_lib) {
        ocl_lib = dlopen("libOpenCL.so", RTLD_NOW);
    }
#endif
    return ocl_lib != NULL;
}
#endif

// --- OpenCL 1.2 ABI subset (values match CL/cl.h) ---

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint32_t cl_bool;
typedef uint64_t cl_device_type;
typedef uint64_t cl_mem_flags;
typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_mem;
typedef void *cl_program;
typedef void *cl_kernel;

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_DEVICE_TYPE_GPU ((cl_device_type)(1 << 2))
#define CL_DEVICE_TYPE_CPU ((cl_device_type)(1 << 1))
#define CL_DEVICE_TYPE_ACCELERATOR ((cl_device_type)(1 << 3))
#define CL_MEM_READ_WRITE ((cl_mem_flags)(1 << 0))
#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_AVAILABLE 0x1027
#define CL_DEVICE_COMPILER_AVAILABLE 0x1028
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_DOUBLE_FP_CONFIG 0x1032
#define CL_PROGRAM_BUILD_LOG 0x1183

typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_context (*fn_clCreateContext)(const void *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
typedef cl_command_queue (*fn_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int *);
typedef cl_program (*fn_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (*fn_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
typedef cl_kernel (*fn_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (*fn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_mem (*fn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_int (*fn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const void *, void *);
typedef cl_int (*fn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const void *, void *);
typedef cl_int (*fn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const void *, void *);
typedef cl_int (*fn_clFinish)(cl_command_queue);
typedef cl_int (*fn_clWaitForEvents)(cl_uint, const void *);
typedef cl_int (*fn_clReleaseEvent)(void *);
typedef cl_int (*fn_clReleaseMemObject)(cl_mem);
typedef cl_int (*fn_clReleaseKernel)(cl_kernel);
typedef cl_int (*fn_clReleaseProgram)(cl_program);
typedef cl_int (*fn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*fn_clReleaseContext)(cl_context);
typedef cl_int (*fn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
typedef cl_int (*fn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);

static fn_clGetPlatformIDs p_clGetPlatformIDs = 0;
static fn_clGetDeviceIDs p_clGetDeviceIDs = 0;
static fn_clCreateContext p_clCreateContext = 0;
static fn_clCreateCommandQueue p_clCreateCommandQueue = 0;
static fn_clCreateProgramWithSource p_clCreateProgramWithSource = 0;
static fn_clBuildProgram p_clBuildProgram = 0;
static fn_clCreateKernel p_clCreateKernel = 0;
static fn_clSetKernelArg p_clSetKernelArg = 0;
static fn_clCreateBuffer p_clCreateBuffer = 0;
static fn_clEnqueueWriteBuffer p_clEnqueueWriteBuffer = 0;
static fn_clEnqueueReadBuffer p_clEnqueueReadBuffer = 0;
static fn_clEnqueueNDRangeKernel p_clEnqueueNDRangeKernel = 0;
static fn_clFinish p_clFinish = 0;
static fn_clWaitForEvents p_clWaitForEvents = 0;
static fn_clReleaseEvent p_clReleaseEvent = 0;
static fn_clReleaseMemObject p_clReleaseMemObject = 0;
static fn_clReleaseKernel p_clReleaseKernel = 0;
static fn_clReleaseProgram p_clReleaseProgram = 0;
static fn_clReleaseCommandQueue p_clReleaseCommandQueue = 0;
static fn_clReleaseContext p_clReleaseContext = 0;
static fn_clGetDeviceInfo p_clGetDeviceInfo = 0;
static fn_clGetProgramBuildInfo p_clGetProgramBuildInfo = 0;

#define OCL_WANT(name) \
    p_##name = (fn_##name)ocl_sym(#name); \
    if (!p_##name) return 0;

// --- Kernel sources (OpenCL C 1.2) ---

static const char *ocl_kernel_src =
    "__kernel void tadd_f32(__global const float* a, __global const float* b, __global float* o, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) o[i] = a[i] + b[i];\n"
    "}\n"
    "__kernel void tmul_f32(__global const float* a, __global const float* b, __global float* o, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) o[i] = a[i] * b[i];\n"
    "}\n"
    "__kernel void tadd_i32(__global const int* a, __global const int* b, __global int* o, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) o[i] = a[i] + b[i];\n"
    "}\n"
    "__kernel void tmul_i32(__global const int* a, __global const int* b, __global int* o, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) o[i] = a[i] * b[i];\n"
    "}\n"
    "__kernel void tadd_i64(__global const long* a, __global const long* b, __global long* o, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) o[i] = a[i] + b[i];\n"
    "}\n"
    "__kernel void tmul_i64(__global const long* a, __global const long* b, __global long* o, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) o[i] = a[i] * b[i];\n"
    "}\n"
    "__kernel void tmm_f32(__global const float* A, __global const float* B, __global float* C,\n"
    "                      int ao, int bo, int ro, int M, int N, int K) {\n"
    "    int j = get_global_id(0);\n"
    "    int i = get_global_id(1);\n"
    "    int lj = get_local_id(0);\n"
    "    int li = get_local_id(1);\n"
    "    __local float As[16][16];\n"
    "    __local float Bs[16][16];\n"
    "    float s = 0.0f;\n"
    "    int tiles = (K + 15) / 16;\n"
    "    for (int t = 0; t < tiles; ++t) {\n"
    "        int pa = t * 16 + lj;\n"
    "        int pb = t * 16 + li;\n"
    "        As[li][lj] = (i < M && pa < K) ? A[ao + i * K + pa] : 0.0f;\n"
    "        Bs[li][lj] = (pb < K && j < N) ? B[bo + pb * N + j] : 0.0f;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int p = 0; p < 16; ++p) s += As[li][p] * Bs[p][lj];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (i < M && j < N) C[ro + i * N + j] = s;\n"
    "}\n"
    "__kernel void tmm_i32(__global const int* A, __global const int* B, __global int* C,\n"
    "                      int ao, int bo, int ro, int M, int N, int K) {\n"
    "    int j = get_global_id(0);\n"
    "    int i = get_global_id(1);\n"
    "    int lj = get_local_id(0);\n"
    "    int li = get_local_id(1);\n"
    "    __local int As[16][16];\n"
    "    __local int Bs[16][16];\n"
    "    int s = 0;\n"
    "    int tiles = (K + 15) / 16;\n"
    "    for (int t = 0; t < tiles; ++t) {\n"
    "        int pa = t * 16 + lj;\n"
    "        int pb = t * 16 + li;\n"
    "        As[li][lj] = (i < M && pa < K) ? A[ao + i * K + pa] : 0;\n"
    "        Bs[li][lj] = (pb < K && j < N) ? B[bo + pb * N + j] : 0;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int p = 0; p < 16; ++p) s += As[li][p] * Bs[p][lj];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (i < M && j < N) C[ro + i * N + j] = s;\n"
    "}\n"
    "__kernel void tmm_i64(__global const long* A, __global const long* B, __global long* C,\n"
    "                      int ao, int bo, int ro, int M, int N, int K) {\n"
    "    int j = get_global_id(0);\n"
    "    int i = get_global_id(1);\n"
    "    int lj = get_local_id(0);\n"
    "    int li = get_local_id(1);\n"
    "    __local long As[16][16];\n"
    "    __local long Bs[16][16];\n"
    "    long s = 0;\n"
    "    int tiles = (K + 15) / 16;\n"
    "    for (int t = 0; t < tiles; ++t) {\n"
    "        int pa = t * 16 + lj;\n"
    "        int pb = t * 16 + li;\n"
    "        As[li][lj] = (i < M && pa < K) ? A[ao + i * K + pa] : 0L;\n"
    "        Bs[li][lj] = (pb < K && j < N) ? B[bo + pb * N + j] : 0L;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int p = 0; p < 16; ++p) s += As[li][p] * Bs[p][lj];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (i < M && j < N) C[ro + i * N + j] = s;\n"
    "}\n"
    "#ifdef FP64\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "__kernel void tadd_f64(__global const double* a, __global const double* b, __global double* o, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) o[i] = a[i] + b[i];\n"
    "}\n"
    "__kernel void tmul_f64(__global const double* a, __global const double* b, __global double* o, int n) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < n) o[i] = a[i] * b[i];\n"
    "}\n"
    "__kernel void tmm_f64(__global const double* A, __global const double* B, __global double* C,\n"
    "                      int ao, int bo, int ro, int M, int N, int K) {\n"
    "    int j = get_global_id(0);\n"
    "    int i = get_global_id(1);\n"
    "    int lj = get_local_id(0);\n"
    "    int li = get_local_id(1);\n"
    "    __local double As[16][16];\n"
    "    __local double Bs[16][16];\n"
    "    double s = 0.0;\n"
    "    int tiles = (K + 15) / 16;\n"
    "    for (int t = 0; t < tiles; ++t) {\n"
    "        int pa = t * 16 + lj;\n"
    "        int pb = t * 16 + li;\n"
    "        As[li][lj] = (i < M && pa < K) ? A[ao + i * K + pa] : 0.0;\n"
    "        Bs[li][lj] = (pb < K && j < N) ? B[bo + pb * N + j] : 0.0;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int p = 0; p < 16; ++p) s += As[li][p] * Bs[p][lj];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (i < M && j < N) C[ro + i * N + j] = s;\n"
    "}\n"
    "#endif\n";

static const char *ocl_kernel_names[12] = {
    "tadd_f32", "tmul_f32", "tadd_f64", "tmul_f64",
    "tadd_i32", "tmul_i32", "tadd_i64", "tmul_i64",
    "tmm_f32", "tmm_f64", "tmm_i32", "tmm_i64"
};

// --- Backend state (single device, process lifetime) ---

static int ocl_state = 0; // 0 = unprobed, 1 = ready, -1 = unavailable
static int ocl_device_count = 0;
static int ocl_has_fp64 = 0;
static cl_context ocl_ctx = 0;
static cl_command_queue ocl_queue = 0;
static cl_device_id ocl_device = 0;
static cl_kernel ocl_kernels[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static char ocl_error[2048] = {0};
static char ocl_dev_name[256] = {0};

// In-flight launch events for the async fence. Launches enqueue without
// blocking; the in-order queue preserves submission order and blocking
// reads serve as implicit barriers, so `sync` only needs to drain.
#define OCL_MAX_EVENTS 64
static void *ocl_events[OCL_MAX_EVENTS] = {0};
static int ocl_event_count = 0;

static void ocl_track(void *ev) {
    int i = 0;
    if (!ev) return;
    if (ocl_event_count >= OCL_MAX_EVENTS) {
        // Ring full: drain before tracking more (preserves correctness,
        // keeps memory bounded).
        p_clFinish(ocl_queue);
        for (i = 0; i < ocl_event_count; ++i) {
            p_clReleaseEvent(ocl_events[i]);
            ocl_events[i] = 0;
        }
        ocl_event_count = 0;
    }
    ocl_events[ocl_event_count++] = ev;
}

static void ocl_trace(const char *what) {
    const char *on = getenv("ALYA_TENSOR_OCL_DEBUG");
    if (on && on[0]) {
        fprintf(stderr, "[tensor-ocl] %s\n", what);
    }
}

static void ocl_fail(const char *msg) {
    size_t n = 0;
    while (n + 1 < sizeof(ocl_error) && msg[n]) {
        ocl_error[n] = msg[n];
        ++n;
    }
    ocl_error[n] = 0;
}

// A failed launch/transfer/sync means the context is likely dead: poison
// the backend for the process lifetime so later calls fail safe (CPU
// fallback) instead of blocking forever on a wedged context.
static void ocl_poison(void) {
    ocl_state = -1;
}

static int ocl_load_api(void) {
    if (!ocl_lib_open()) return 0;
    OCL_WANT(clGetPlatformIDs);
    OCL_WANT(clGetDeviceIDs);
    OCL_WANT(clCreateContext);
    OCL_WANT(clCreateCommandQueue);
    OCL_WANT(clCreateProgramWithSource);
    OCL_WANT(clBuildProgram);
    OCL_WANT(clCreateKernel);
    OCL_WANT(clSetKernelArg);
    OCL_WANT(clCreateBuffer);
    OCL_WANT(clEnqueueWriteBuffer);
    OCL_WANT(clEnqueueReadBuffer);
    OCL_WANT(clEnqueueNDRangeKernel);
    OCL_WANT(clFinish);
    OCL_WANT(clWaitForEvents);
    OCL_WANT(clReleaseEvent);
    OCL_WANT(clReleaseMemObject);
    OCL_WANT(clReleaseKernel);
    OCL_WANT(clReleaseProgram);
    OCL_WANT(clReleaseCommandQueue);
    OCL_WANT(clReleaseContext);
    OCL_WANT(clGetDeviceInfo);
    OCL_WANT(clGetProgramBuildInfo);
    return 1;
}

static int ocl_pick_device(cl_platform_id plat, cl_device_type type, cl_device_id *out, cl_uint *count) {
    cl_uint n = 0;
    if (p_clGetDeviceIDs(plat, type, 0, 0, &n) != CL_SUCCESS || n == 0) return 0;
    if (p_clGetDeviceIDs(plat, type, 1, out, 0) != CL_SUCCESS) return 0;
    *count = n;
    return 1;
}

static void ocl_init(void) {
    cl_platform_id plats[8];
    cl_uint nplat = 0;
    cl_device_id dev = 0;
    cl_uint ndev = 0;
    cl_int err = 0;
    cl_program prog = 0;
    int ki = 0;

    if (ocl_state != 0) return;
    ocl_state = -1;
    if (!ocl_load_api()) {
        ocl_fail("OpenCL loader library not found");
        return;
    }
    if (p_clGetPlatformIDs(8, plats, &nplat) != CL_SUCCESS || nplat == 0) {
        ocl_fail("no OpenCL platform");
        return;
    }
    {
        cl_uint pi = 0;
        int found = 0;
        for (pi = 0; pi < nplat; ++pi) {
            if (ocl_pick_device(plats[pi], CL_DEVICE_TYPE_GPU, &dev, &ndev)) {
                found = 1;
                break;
            }
        }
        if (!found) {
            for (pi = 0; pi < nplat; ++pi) {
                if (ocl_pick_device(plats[pi], CL_DEVICE_TYPE_CPU, &dev, &ndev)) {
                    found = 1;
                    break;
                }
            }
        }
        if (!found) {
            ocl_fail("no OpenCL GPU/CPU device");
            return;
        }
    }
    {
        cl_bool avail = 0;
        if (p_clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(avail), &avail, 0) != CL_SUCCESS || !avail) {
            ocl_fail("OpenCL device not available");
            return;
        }
    }
    {
        cl_ulong fp64 = 0;
        if (p_clGetDeviceInfo(dev, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, 0) == CL_SUCCESS && fp64) {
            ocl_has_fp64 = 1;
        }
    }
    ocl_ctx = p_clCreateContext(0, 1, &dev, 0, 0, &err);
    if (!ocl_ctx || err != CL_SUCCESS) {
        ocl_fail("clCreateContext failed");
        return;
    }
    ocl_queue = p_clCreateCommandQueue(ocl_ctx, dev, 0, &err);
    if (!ocl_queue || err != CL_SUCCESS) {
        ocl_fail("clCreateCommandQueue failed");
        return;
    }
    prog = p_clCreateProgramWithSource(ocl_ctx, 1, &ocl_kernel_src, 0, &err);
    if (!prog || err != CL_SUCCESS) {
        ocl_fail("clCreateProgramWithSource failed");
        return;
    }
    err = p_clBuildProgram(prog, 1, &dev, ocl_has_fp64 ? "-DFP64" : "", 0, 0);
    if (err != CL_SUCCESS) {
        char log[1024] = {0};
        size_t got = 0;
        p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, &got);
        (void)got;
        ocl_fail(log[0] ? log : "clBuildProgram failed");
        p_clReleaseProgram(prog);
        return;
    }
    for (ki = 0; ki < 12; ++ki) {
        if ((ki == 2 || ki == 3 || ki == 9) && !ocl_has_fp64) continue;
        ocl_kernels[ki] = p_clCreateKernel(prog, ocl_kernel_names[ki], &err);
        if (!ocl_kernels[ki] || err != CL_SUCCESS) {
            ocl_kernels[ki] = 0;
        }
    }
    p_clReleaseProgram(prog);
    ocl_device = dev;
    ocl_device_count = (int)ndev;
    ocl_state = 1;
    if (p_clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(ocl_dev_name) - 1, ocl_dev_name, 0) != CL_SUCCESS) {
        ocl_dev_name[0] = 0;
    }
    {
        char ready[300] = {0};
        const char *prefix = "backend ready on ";
        size_t pi = 0;
        while (pi + 1 < sizeof(ready) && prefix[pi]) {
            ready[pi] = prefix[pi];
            ++pi;
        }
        {
            size_t di = 0;
            while (pi + 1 < sizeof(ready) && ocl_dev_name[di]) {
                ready[pi] = ocl_dev_name[di];
                ++pi;
                ++di;
            }
        }
        ready[pi] = 0;
        ocl_trace(ready);
    }
}

// Kernel table index by (op, dtype): op 0 = add, 1 = mul, 2 = matmul;
// dtype 0 = f64, 1 = f32, 2 = i32, 3 = i64.
static int ocl_kernel_index(int op, int dtype) {
    static const int table[3][4] = {
        {2, 0, 4, 6},
        {3, 1, 5, 7},
        {9, 8, 10, 11}
    };
    if (op < 0 || op > 2 || dtype < 0 || dtype > 3) return -1;
    return table[op][dtype];
}

// --- Public entry points (called from device.c and the Alya FFI) ---

int32_t alya_tensor_ocl_count(void) {
    ocl_init();
    if (ocl_state != 1) return 0;
    return (int32_t)ocl_device_count;
}

int32_t alya_tensor_ocl_supports(int32_t dtype) {
    ocl_init();
    if (ocl_state != 1) return 0;
    if (dtype == 0) return ocl_has_fp64 ? 1 : 0;
    if (dtype == 1 || dtype == 2 || dtype == 3) return 1;
    return 0;
}

const char *alya_tensor_ocl_error(void) {
    return ocl_error;
}

const char *alya_tensor_ocl_name(void) {
    ocl_init();
    return ocl_dev_name;
}

void *alya_tensor_ocl_alloc(int32_t byte_size) {
    cl_int err = 0;
    cl_mem m = 0;
    ocl_init();
    if (ocl_state != 1 || byte_size <= 0) return 0;
    m = p_clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, (size_t)byte_size, 0, &err);
    if (!m || err != CL_SUCCESS) return 0;
    return (void *)m;
}

void alya_tensor_ocl_free(void *handle) {
    if (handle) p_clReleaseMemObject((cl_mem)handle);
}

int32_t alya_tensor_ocl_write(void *handle, void *host, int32_t byte_size) {
    ocl_init();
    if (ocl_state != 1 || !handle || !host || byte_size <= 0) return 0;
    if (p_clEnqueueWriteBuffer(ocl_queue, (cl_mem)handle, CL_TRUE, 0, (size_t)byte_size, host, 0, 0, 0) != CL_SUCCESS) {
        return 0;
    }
    return 1;
}

int32_t alya_tensor_ocl_read(void *handle, void *host, int32_t byte_size) {
    ocl_init();
    if (ocl_state != 1 || !handle || !host || byte_size <= 0) return 0;
    if (p_clEnqueueReadBuffer(ocl_queue, (cl_mem)handle, CL_TRUE, 0, (size_t)byte_size, host, 0, 0, 0) != CL_SUCCESS) {
        return 0;
    }
    return 1;
}

static int32_t ocl_launch_ew(void *ah, void *bh, void *oh, int32_t count, int op, int32_t dtype) {
    int ki = 0;
    cl_kernel k = 0;
    size_t global = 0;
    ocl_init();
    if (ocl_state != 1 || !ah || !bh || !oh || count <= 0) return 0;
    ki = ocl_kernel_index(op, (int)dtype);
    if (ki < 0 || !ocl_kernels[ki]) return 0;
    k = ocl_kernels[ki];
    if (p_clSetKernelArg(k, 0, sizeof(void *), &ah) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(k, 1, sizeof(void *), &bh) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(k, 2, sizeof(void *), &oh) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(k, 3, sizeof(int32_t), &count) != CL_SUCCESS) return 0;
    global = (size_t)count;
    {
        void *ev = 0;
        if (p_clEnqueueNDRangeKernel(ocl_queue, k, 1, 0, &global, 0, 0, 0, &ev) != CL_SUCCESS) {
            ocl_poison();
            return 0;
        }
        ocl_track(ev);
    }
    ocl_trace("element-wise launch ok");
    return 1;
}

int32_t alya_tensor_ocl_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    return ocl_launch_ew(ah, bh, oh, count, 0, dtype);
}

int32_t alya_tensor_ocl_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    return ocl_launch_ew(ah, bh, oh, count, 1, dtype);
}

int32_t alya_tensor_ocl_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off, int32_t r_off,
                               int32_t m, int32_t n, int32_t k, int32_t dtype) {
    int ki = 0;
    cl_kernel kr = 0;
    ocl_init();
    if (ocl_state != 1 || !ah || !bh || !oh || m <= 0 || n <= 0 || k <= 0) return 0;
    ki = ocl_kernel_index(2, (int)dtype);
    if (ki < 0 || !ocl_kernels[ki]) return 0;
    kr = ocl_kernels[ki];
    if (p_clSetKernelArg(kr, 0, sizeof(void *), &ah) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(kr, 1, sizeof(void *), &bh) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(kr, 2, sizeof(void *), &oh) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(kr, 3, sizeof(int32_t), &a_off) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(kr, 4, sizeof(int32_t), &b_off) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(kr, 5, sizeof(int32_t), &r_off) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(kr, 6, sizeof(int32_t), &m) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(kr, 7, sizeof(int32_t), &n) != CL_SUCCESS) return 0;
    if (p_clSetKernelArg(kr, 8, sizeof(int32_t), &k) != CL_SUCCESS) return 0;
    {
        // Tiled kernels run in fixed 16x16 work-groups: round the range up
        // (out-of-range threads exit via bounds checks) and pin local size.
        size_t rounded[2];
        size_t local[2];
        rounded[0] = ((size_t)n + 15) / 16 * 16;
        rounded[1] = ((size_t)m + 15) / 16 * 16;
        local[0] = 16;
        local[1] = 16;
        {
            void *ev = 0;
            if (p_clEnqueueNDRangeKernel(ocl_queue, kr, 2, 0, rounded, local, 0, 0, &ev) != CL_SUCCESS) {
                ocl_poison();
                return 0;
            }
            ocl_track(ev);
        }
    }
    ocl_trace("matmul launch ok");
    return 1;
}

int32_t alya_tensor_ocl_sync(void) {
    int i = 0;
    int ok = 1;
    ocl_init();
    if (ocl_state != 1) return 0;
    if (ocl_event_count > 0) {
        if (p_clWaitForEvents((cl_uint)ocl_event_count, (const void *)ocl_events) != CL_SUCCESS) {
            ok = 0;
        }
        for (i = 0; i < ocl_event_count; ++i) {
            p_clReleaseEvent(ocl_events[i]);
            ocl_events[i] = 0;
        }
        ocl_event_count = 0;
    }
    if (p_clFinish(ocl_queue) != CL_SUCCESS) ok = 0;
    if (!ok) ocl_poison();
    return (int32_t)ok;
}
