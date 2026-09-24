// tensor CUDA compute backend (portable, dynamically loaded PTX).
//
// Part of the Alya Language package ecosystem (https://github.com/alya-lang).
//
// No CUDA toolkit is required: the driver API (nvcuda.dll / libcuda) is
// loaded at runtime with hand-declared prototypes, and the 12 kernels are
// embedded as PTX 6.0 / sm_50 source strings JIT-compiled by the driver
// via cuModuleLoadData. Devices below sm_50 are declined (module load
// would fail); with no driver present every entry reports unavailable and
// the Alya side falls back. f64 is first-class here (unlike Apple GPUs).
// Compiles anywhere (`gcc -fsyntax-only` passes); executes on NVIDIA GPUs.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
static HMODULE cuda_lib = NULL;
static void *cuda_sym(const char *name) {
    return (void *)GetProcAddress(cuda_lib, name);
}
static int cuda_lib_open(void) {
    cuda_lib = LoadLibraryA("nvcuda.dll");
    return cuda_lib != NULL;
}
#else
#include <dlfcn.h>
static void *cuda_lib = NULL;
static void *cuda_sym(const char *name) {
    return dlsym(cuda_lib, name);
}
static int cuda_lib_open(void) {
    cuda_lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!cuda_lib) {
        cuda_lib = dlopen("libcuda.so", RTLD_NOW);
    }
    if (!cuda_lib) {
        cuda_lib = dlopen("libcuda.dylib", RTLD_NOW);
    }
    return cuda_lib != NULL;
}
#endif

// --- CUDA driver API subset (values match cuda.h) ---

typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

typedef int (*fn_cuInit)(unsigned int);
typedef int (*fn_cuDeviceGetCount)(int *);
typedef int (*fn_cuDeviceGet)(CUdevice *, int);
typedef int (*fn_cuDeviceGetName)(char *, int, CUdevice);
typedef int (*fn_cuDeviceGetAttribute)(int *, int, CUdevice);
typedef int (*fn_cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice);
typedef int (*fn_cuModuleLoadData)(CUmodule *, const void *);
typedef int (*fn_cuModuleLoadDataEx)(CUmodule *, const void *, unsigned int, int *, void **);
typedef int (*fn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef int (*fn_cuMemAlloc_v2)(CUdeviceptr *, size_t);
typedef int (*fn_cuMemFree_v2)(CUdeviceptr);
typedef int (*fn_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
typedef int (*fn_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
typedef int (*fn_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                 unsigned int, unsigned int, unsigned int,
                                 unsigned int, CUstream, void **, void *);
typedef int (*fn_cuCtxSynchronize)(void);

static fn_cuInit p_cuInit = 0;
static fn_cuDeviceGetCount p_cuDeviceGetCount = 0;
static fn_cuDeviceGet p_cuDeviceGet = 0;
static fn_cuDeviceGetName p_cuDeviceGetName = 0;
static fn_cuDeviceGetAttribute p_cuDeviceGetAttribute = 0;
static fn_cuCtxCreate_v2 p_cuCtxCreate_v2 = 0;
static fn_cuModuleLoadData p_cuModuleLoadData = 0;
static fn_cuModuleLoadDataEx p_cuModuleLoadDataEx = 0;
static fn_cuModuleGetFunction p_cuModuleGetFunction = 0;
static fn_cuMemAlloc_v2 p_cuMemAlloc_v2 = 0;
static fn_cuMemFree_v2 p_cuMemFree_v2 = 0;
static fn_cuMemcpyHtoD_v2 p_cuMemcpyHtoD_v2 = 0;
static fn_cuMemcpyDtoH_v2 p_cuMemcpyDtoH_v2 = 0;
static fn_cuLaunchKernel p_cuLaunchKernel = 0;
static fn_cuCtxSynchronize p_cuCtxSynchronize = 0;

#define CUDA_WANT(name) \
    p_##name = (fn_##name)cuda_sym(#name); \
    if (!p_##name) return 0;

// --- PTX kernels (6.0 / sm_50; driver JIT-compiles forward) ---

static const char *cuda_ptx_src =
    ".version 6.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    ".visible .entry tadd_f32(.param .u64 a, .param .u64 b, .param .u64 o, .param .u32 n)\n"
    "{\n"
    "    .reg .u64 %rd1, %rd2, %rd3, %rd4;\n"
    "    .reg .u32 %r1, %r2, %r3, %r4;\n"
    "    .reg .pred %p1;\n"
    "    .reg .f32 %f1, %f2, %f3;\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mad.lo.u32 %r1, %r3, %r2, %r1;\n"
    "    ld.param.u32 %r4, [n];\n"
    "    setp.ge.u32 %p1, %r1, %r4;\n"
    "    @%p1 bra $done;\n"
    "    mul.wide.u32 %rd1, %r1, 4;\n"
    "    ld.param.u64 %rd2, [a];\n"
    "    ld.param.u64 %rd3, [b];\n"
    "    ld.param.u64 %rd4, [o];\n"
    "    add.u64 %rd2, %rd2, %rd1;\n"
    "    add.u64 %rd3, %rd3, %rd1;\n"
    "    add.u64 %rd4, %rd4, %rd1;\n"
    "    ld.global.f32 %f1, [%rd2];\n"
    "    ld.global.f32 %f2, [%rd3];\n"
    "    add.f32 %f3, %f1, %f2;\n"
    "    st.global.f32 [%rd4], %f3;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tmul_f32(.param .u64 a, .param .u64 b, .param .u64 o, .param .u32 n)\n"
    "{\n"
    "    .reg .u64 %rd1, %rd2, %rd3, %rd4;\n"
    "    .reg .u32 %r1, %r2, %r3, %r4;\n"
    "    .reg .pred %p1;\n"
    "    .reg .f32 %f1, %f2, %f3;\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mad.lo.u32 %r1, %r3, %r2, %r1;\n"
    "    ld.param.u32 %r4, [n];\n"
    "    setp.ge.u32 %p1, %r1, %r4;\n"
    "    @%p1 bra $done;\n"
    "    mul.wide.u32 %rd1, %r1, 4;\n"
    "    ld.param.u64 %rd2, [a];\n"
    "    ld.param.u64 %rd3, [b];\n"
    "    ld.param.u64 %rd4, [o];\n"
    "    add.u64 %rd2, %rd2, %rd1;\n"
    "    add.u64 %rd3, %rd3, %rd1;\n"
    "    add.u64 %rd4, %rd4, %rd1;\n"
    "    ld.global.f32 %f1, [%rd2];\n"
    "    ld.global.f32 %f2, [%rd3];\n"
    "    mul.f32 %f3, %f1, %f2;\n"
    "    st.global.f32 [%rd4], %f3;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tadd_f64(.param .u64 a, .param .u64 b, .param .u64 o, .param .u32 n)\n"
    "{\n"
    "    .reg .u64 %rd1, %rd2, %rd3, %rd4;\n"
    "    .reg .u32 %r1, %r2, %r3, %r4;\n"
    "    .reg .pred %p1;\n"
    "    .reg .f64 %f1, %f2, %f3;\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mad.lo.u32 %r1, %r3, %r2, %r1;\n"
    "    ld.param.u32 %r4, [n];\n"
    "    setp.ge.u32 %p1, %r1, %r4;\n"
    "    @%p1 bra $done;\n"
    "    mul.wide.u32 %rd1, %r1, 8;\n"
    "    ld.param.u64 %rd2, [a];\n"
    "    ld.param.u64 %rd3, [b];\n"
    "    ld.param.u64 %rd4, [o];\n"
    "    add.u64 %rd2, %rd2, %rd1;\n"
    "    add.u64 %rd3, %rd3, %rd1;\n"
    "    add.u64 %rd4, %rd4, %rd1;\n"
    "    ld.global.f64 %f1, [%rd2];\n"
    "    ld.global.f64 %f2, [%rd3];\n"
    "    add.f64 %f3, %f1, %f2;\n"
    "    st.global.f64 [%rd4], %f3;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tmul_f64(.param .u64 a, .param .u64 b, .param .u64 o, .param .u32 n)\n"
    "{\n"
    "    .reg .u64 %rd1, %rd2, %rd3, %rd4;\n"
    "    .reg .u32 %r1, %r2, %r3, %r4;\n"
    "    .reg .pred %p1;\n"
    "    .reg .f64 %f1, %f2, %f3;\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mad.lo.u32 %r1, %r3, %r2, %r1;\n"
    "    ld.param.u32 %r4, [n];\n"
    "    setp.ge.u32 %p1, %r1, %r4;\n"
    "    @%p1 bra $done;\n"
    "    mul.wide.u32 %rd1, %r1, 8;\n"
    "    ld.param.u64 %rd2, [a];\n"
    "    ld.param.u64 %rd3, [b];\n"
    "    ld.param.u64 %rd4, [o];\n"
    "    add.u64 %rd2, %rd2, %rd1;\n"
    "    add.u64 %rd3, %rd3, %rd1;\n"
    "    add.u64 %rd4, %rd4, %rd1;\n"
    "    ld.global.f64 %f1, [%rd2];\n"
    "    ld.global.f64 %f2, [%rd3];\n"
    "    mul.f64 %f3, %f1, %f2;\n"
    "    st.global.f64 [%rd4], %f3;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tadd_i32(.param .u64 a, .param .u64 b, .param .u64 o, .param .u32 n)\n"
    "{\n"
    "    .reg .u64 %rd1, %rd2, %rd3, %rd4;\n"
    "    .reg .u32 %r1, %r2, %r3, %r4;\n"
    "    .reg .pred %p1;\n"
    "    .reg .s32 %f1, %f2, %f3;\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mad.lo.u32 %r1, %r3, %r2, %r1;\n"
    "    ld.param.u32 %r4, [n];\n"
    "    setp.ge.u32 %p1, %r1, %r4;\n"
    "    @%p1 bra $done;\n"
    "    mul.wide.u32 %rd1, %r1, 4;\n"
    "    ld.param.u64 %rd2, [a];\n"
    "    ld.param.u64 %rd3, [b];\n"
    "    ld.param.u64 %rd4, [o];\n"
    "    add.u64 %rd2, %rd2, %rd1;\n"
    "    add.u64 %rd3, %rd3, %rd1;\n"
    "    add.u64 %rd4, %rd4, %rd1;\n"
    "    ld.global.s32 %f1, [%rd2];\n"
    "    ld.global.s32 %f2, [%rd3];\n"
    "    add.s32 %f3, %f1, %f2;\n"
    "    st.global.s32 [%rd4], %f3;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tmul_i32(.param .u64 a, .param .u64 b, .param .u64 o, .param .u32 n)\n"
    "{\n"
    "    .reg .u64 %rd1, %rd2, %rd3, %rd4;\n"
    "    .reg .u32 %r1, %r2, %r3, %r4;\n"
    "    .reg .pred %p1;\n"
    "    .reg .s32 %f1, %f2, %f3;\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mad.lo.u32 %r1, %r3, %r2, %r1;\n"
    "    ld.param.u32 %r4, [n];\n"
    "    setp.ge.u32 %p1, %r1, %r4;\n"
    "    @%p1 bra $done;\n"
    "    mul.wide.u32 %rd1, %r1, 4;\n"
    "    ld.param.u64 %rd2, [a];\n"
    "    ld.param.u64 %rd3, [b];\n"
    "    ld.param.u64 %rd4, [o];\n"
    "    add.u64 %rd2, %rd2, %rd1;\n"
    "    add.u64 %rd3, %rd3, %rd1;\n"
    "    add.u64 %rd4, %rd4, %rd1;\n"
    "    ld.global.s32 %f1, [%rd2];\n"
    "    ld.global.s32 %f2, [%rd3];\n"
    "    mul.lo.s32 %f3, %f1, %f2;\n"
    "    st.global.s32 [%rd4], %f3;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tadd_i64(.param .u64 a, .param .u64 b, .param .u64 o, .param .u32 n)\n"
    "{\n"
    "    .reg .u64 %rd1, %rd2, %rd3, %rd4;\n"
    "    .reg .u32 %r1, %r2, %r3, %r4;\n"
    "    .reg .pred %p1;\n"
    "    .reg .s64 %f1, %f2, %f3;\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mad.lo.u32 %r1, %r3, %r2, %r1;\n"
    "    ld.param.u32 %r4, [n];\n"
    "    setp.ge.u32 %p1, %r1, %r4;\n"
    "    @%p1 bra $done;\n"
    "    mul.wide.u32 %rd1, %r1, 8;\n"
    "    ld.param.u64 %rd2, [a];\n"
    "    ld.param.u64 %rd3, [b];\n"
    "    ld.param.u64 %rd4, [o];\n"
    "    add.u64 %rd2, %rd2, %rd1;\n"
    "    add.u64 %rd3, %rd3, %rd1;\n"
    "    add.u64 %rd4, %rd4, %rd1;\n"
    "    ld.global.s64 %f1, [%rd2];\n"
    "    ld.global.s64 %f2, [%rd3];\n"
    "    add.s64 %f3, %f1, %f2;\n"
    "    st.global.s64 [%rd4], %f3;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tmul_i64(.param .u64 a, .param .u64 b, .param .u64 o, .param .u32 n)\n"
    "{\n"
    "    .reg .u64 %rd1, %rd2, %rd3, %rd4;\n"
    "    .reg .u32 %r1, %r2, %r3, %r4;\n"
    "    .reg .pred %p1;\n"
    "    .reg .s64 %f1, %f2, %f3;\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %ctaid.x;\n"
    "    mad.lo.u32 %r1, %r3, %r2, %r1;\n"
    "    ld.param.u32 %r4, [n];\n"
    "    setp.ge.u32 %p1, %r1, %r4;\n"
    "    @%p1 bra $done;\n"
    "    mul.wide.u32 %rd1, %r1, 8;\n"
    "    ld.param.u64 %rd2, [a];\n"
    "    ld.param.u64 %rd3, [b];\n"
    "    ld.param.u64 %rd4, [o];\n"
    "    add.u64 %rd2, %rd2, %rd1;\n"
    "    add.u64 %rd3, %rd3, %rd1;\n"
    "    add.u64 %rd4, %rd4, %rd1;\n"
    "    ld.global.s64 %f1, [%rd2];\n"
    "    ld.global.s64 %f2, [%rd3];\n"
    "    mul.lo.s64 %f3, %f1, %f2;\n"
    "    st.global.s64 [%rd4], %f3;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tmm_f32(.param .u64 A, .param .u64 B, .param .u64 C,\n"
    "    .param .u32 ao, .param .u32 bo, .param .u32 ro,\n"
    "    .param .u32 M, .param .u32 N, .param .u32 K)\n"
    "{\n"
    "    .reg .u64 %rdA, %rdB, %rdC, %off;\n"
    "    .reg .u32 %i, %j, %p, %M, %N, %K, %ao, %bo, %ro, %t, %cx, %cy;\n"
    "    .reg .pred %pM, %pN, %pK;\n"
    "    .reg .f32 %s, %x, %y;\n"
    "    mov.u32 %i, %tid.y;\n"
    "    mov.u32 %t, %ntid.y;\n"
    "    mov.u32 %j, %tid.x;\n"
    "    mov.u32 %cy, %ctaid.y;\n"
    "    mad.lo.u32 %i, %cy, %t, %i;\n"
    "    mov.u32 %t, %ntid.x;\n"
    "    mov.u32 %cx, %ctaid.x;\n"
    "    mad.lo.u32 %j, %cx, %t, %j;\n"
    "    ld.param.u32 %M, [M];\n"
    "    ld.param.u32 %N, [N];\n"
    "    ld.param.u32 %K, [K];\n"
    "    ld.param.u32 %ao, [ao];\n"
    "    ld.param.u32 %bo, [bo];\n"
    "    ld.param.u32 %ro, [ro];\n"
    "    ld.param.u64 %rdA, [A];\n"
    "    ld.param.u64 %rdB, [B];\n"
    "    ld.param.u64 %rdC, [C];\n"
    "    setp.ge.u32 %pM, %i, %M;\n"
    "    @%pM bra $done;\n"
    "    setp.ge.u32 %pN, %j, %N;\n"
    "    @%pN bra $done;\n"
    "    mov.f32 %s, 0F00000000;\n"
    "    mov.u32 %p, 0;\n"
    "$loop:\n"
    "    setp.ge.u32 %pK, %p, %K;\n"
    "    @%pK bra $store;\n"
    "    mad.lo.u32 %t, %i, %K, %p;\n"
    "    add.u32 %t, %t, %ao;\n"
    "    mul.wide.u32 %off, %t, 4;\n"
    "    add.u64 %off, %rdA, %off;\n"
    "    ld.global.f32 %x, [%off];\n"
    "    mad.lo.u32 %t, %p, %N, %j;\n"
    "    add.u32 %t, %t, %bo;\n"
    "    mul.wide.u32 %off, %t, 4;\n"
    "    add.u64 %off, %rdB, %off;\n"
    "    ld.global.f32 %y, [%off];\n"
    "    fma.rn.f32 %s, %x, %y, %s;\n"
    "    add.u32 %p, %p, 1;\n"
    "    bra $loop;\n"
    "$store:\n"
    "    mad.lo.u32 %t, %i, %N, %j;\n"
    "    add.u32 %t, %t, %ro;\n"
    "    mul.wide.u32 %off, %t, 4;\n"
    "    add.u64 %off, %rdC, %off;\n"
    "    st.global.f32 [%off], %s;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tmm_f64(.param .u64 A, .param .u64 B, .param .u64 C,\n"
    "    .param .u32 ao, .param .u32 bo, .param .u32 ro,\n"
    "    .param .u32 M, .param .u32 N, .param .u32 K)\n"
    "{\n"
    "    .reg .u64 %rdA, %rdB, %rdC, %off;\n"
    "    .reg .u32 %i, %j, %p, %M, %N, %K, %ao, %bo, %ro, %t, %cx, %cy;\n"
    "    .reg .pred %pM, %pN, %pK;\n"
    "    .reg .f64 %s, %x, %y;\n"
    "    mov.u32 %i, %tid.y;\n"
    "    mov.u32 %t, %ntid.y;\n"
    "    mov.u32 %j, %tid.x;\n"
    "    mov.u32 %cy, %ctaid.y;\n"
    "    mad.lo.u32 %i, %cy, %t, %i;\n"
    "    mov.u32 %t, %ntid.x;\n"
    "    mov.u32 %cx, %ctaid.x;\n"
    "    mad.lo.u32 %j, %cx, %t, %j;\n"
    "    ld.param.u32 %M, [M];\n"
    "    ld.param.u32 %N, [N];\n"
    "    ld.param.u32 %K, [K];\n"
    "    ld.param.u32 %ao, [ao];\n"
    "    ld.param.u32 %bo, [bo];\n"
    "    ld.param.u32 %ro, [ro];\n"
    "    ld.param.u64 %rdA, [A];\n"
    "    ld.param.u64 %rdB, [B];\n"
    "    ld.param.u64 %rdC, [C];\n"
    "    setp.ge.u32 %pM, %i, %M;\n"
    "    @%pM bra $done;\n"
    "    setp.ge.u32 %pN, %j, %N;\n"
    "    @%pN bra $done;\n"
    "    mov.f64 %s, 0D0000000000000000;\n"
    "    mov.u32 %p, 0;\n"
    "$loop:\n"
    "    setp.ge.u32 %pK, %p, %K;\n"
    "    @%pK bra $store;\n"
    "    mad.lo.u32 %t, %i, %K, %p;\n"
    "    add.u32 %t, %t, %ao;\n"
    "    mul.wide.u32 %off, %t, 8;\n"
    "    add.u64 %off, %rdA, %off;\n"
    "    ld.global.f64 %x, [%off];\n"
    "    mad.lo.u32 %t, %p, %N, %j;\n"
    "    add.u32 %t, %t, %bo;\n"
    "    mul.wide.u32 %off, %t, 8;\n"
    "    add.u64 %off, %rdB, %off;\n"
    "    ld.global.f64 %y, [%off];\n"
    "    fma.rn.f64 %s, %x, %y, %s;\n"
    "    add.u32 %p, %p, 1;\n"
    "    bra $loop;\n"
    "$store:\n"
    "    mad.lo.u32 %t, %i, %N, %j;\n"
    "    add.u32 %t, %t, %ro;\n"
    "    mul.wide.u32 %off, %t, 8;\n"
    "    add.u64 %off, %rdC, %off;\n"
    "    st.global.f64 [%off], %s;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tmm_i32(.param .u64 A, .param .u64 B, .param .u64 C,\n"
    "    .param .u32 ao, .param .u32 bo, .param .u32 ro,\n"
    "    .param .u32 M, .param .u32 N, .param .u32 K)\n"
    "{\n"
    "    .reg .u64 %rdA, %rdB, %rdC, %off;\n"
    "    .reg .u32 %i, %j, %p, %M, %N, %K, %ao, %bo, %ro, %t, %cx, %cy;\n"
    "    .reg .pred %pM, %pN, %pK;\n"
    "    .reg .s32 %s, %x, %y;\n"
    "    mov.u32 %i, %tid.y;\n"
    "    mov.u32 %t, %ntid.y;\n"
    "    mov.u32 %j, %tid.x;\n"
    "    mov.u32 %cy, %ctaid.y;\n"
    "    mad.lo.u32 %i, %cy, %t, %i;\n"
    "    mov.u32 %t, %ntid.x;\n"
    "    mov.u32 %cx, %ctaid.x;\n"
    "    mad.lo.u32 %j, %cx, %t, %j;\n"
    "    ld.param.u32 %M, [M];\n"
    "    ld.param.u32 %N, [N];\n"
    "    ld.param.u32 %K, [K];\n"
    "    ld.param.u32 %ao, [ao];\n"
    "    ld.param.u32 %bo, [bo];\n"
    "    ld.param.u32 %ro, [ro];\n"
    "    ld.param.u64 %rdA, [A];\n"
    "    ld.param.u64 %rdB, [B];\n"
    "    ld.param.u64 %rdC, [C];\n"
    "    setp.ge.u32 %pM, %i, %M;\n"
    "    @%pM bra $done;\n"
    "    setp.ge.u32 %pN, %j, %N;\n"
    "    @%pN bra $done;\n"
    "    mov.s32 %s, 0;\n"
    "    mov.u32 %p, 0;\n"
    "    mov.u32 %t, 0;\n"
    "$loop:\n"
    "    setp.ge.u32 %pK, %p, %K;\n"
    "    @%pK bra $store;\n"
    "    mad.lo.u32 %t, %i, %K, %p;\n"
    "    add.u32 %t, %t, %ao;\n"
    "    mul.wide.u32 %off, %t, 4;\n"
    "    add.u64 %off, %rdA, %off;\n"
    "    ld.global.s32 %x, [%off];\n"
    "    mad.lo.u32 %t, %p, %N, %j;\n"
    "    add.u32 %t, %t, %bo;\n"
    "    mul.wide.u32 %off, %t, 4;\n"
    "    add.u64 %off, %rdB, %off;\n"
    "    ld.global.s32 %y, [%off];\n"
    "    mad.lo.s32 %s, %x, %y, %s;\n"
    "    add.u32 %p, %p, 1;\n"
    "    bra $loop;\n"
    "$store:\n"
    "    mad.lo.u32 %t, %i, %N, %j;\n"
    "    add.u32 %t, %t, %ro;\n"
    "    mul.wide.u32 %off, %t, 4;\n"
    "    add.u64 %off, %rdC, %off;\n"
    "    st.global.s32 [%off], %s;\n"
    "$done:\n"
    "    ret;\n"
    "}\n"
    ".visible .entry tmm_i64(.param .u64 A, .param .u64 B, .param .u64 C,\n"
    "    .param .u32 ao, .param .u32 bo, .param .u32 ro,\n"
    "    .param .u32 M, .param .u32 N, .param .u32 K)\n"
    "{\n"
    "    .reg .u64 %rdA, %rdB, %rdC, %off;\n"
    "    .reg .u32 %i, %j, %p, %M, %N, %K, %ao, %bo, %ro, %t, %cx, %cy;\n"
    "    .reg .pred %pM, %pN, %pK;\n"
    "    .reg .s64 %s, %x, %y;\n"
    "    mov.u32 %i, %tid.y;\n"
    "    mov.u32 %t, %ntid.y;\n"
    "    mov.u32 %j, %tid.x;\n"
    "    mov.u32 %cy, %ctaid.y;\n"
    "    mad.lo.u32 %i, %cy, %t, %i;\n"
    "    mov.u32 %t, %ntid.x;\n"
    "    mov.u32 %cx, %ctaid.x;\n"
    "    mad.lo.u32 %j, %cx, %t, %j;\n"
    "    ld.param.u32 %M, [M];\n"
    "    ld.param.u32 %N, [N];\n"
    "    ld.param.u32 %K, [K];\n"
    "    ld.param.u32 %ao, [ao];\n"
    "    ld.param.u32 %bo, [bo];\n"
    "    ld.param.u32 %ro, [ro];\n"
    "    ld.param.u64 %rdA, [A];\n"
    "    ld.param.u64 %rdB, [B];\n"
    "    ld.param.u64 %rdC, [C];\n"
    "    setp.ge.u32 %pM, %i, %M;\n"
    "    @%pM bra $done;\n"
    "    setp.ge.u32 %pN, %j, %N;\n"
    "    @%pN bra $done;\n"
    "    mov.s64 %s, 0;\n"
    "    mov.u32 %p, 0;\n"
    "$loop:\n"
    "    setp.ge.u32 %pK, %p, %K;\n"
    "    @%pK bra $store;\n"
    "    mad.lo.u32 %t, %i, %K, %p;\n"
    "    add.u32 %t, %t, %ao;\n"
    "    mul.wide.u32 %off, %t, 8;\n"
    "    add.u64 %off, %rdA, %off;\n"
    "    ld.global.s64 %x, [%off];\n"
    "    mad.lo.u32 %t, %p, %N, %j;\n"
    "    add.u32 %t, %t, %bo;\n"
    "    mul.wide.u32 %off, %t, 8;\n"
    "    add.u64 %off, %rdB, %off;\n"
    "    ld.global.s64 %y, [%off];\n"
    "    mad.lo.s64 %s, %x, %y, %s;\n"
    "    add.u32 %p, %p, 1;\n"
    "    bra $loop;\n"
    "$store:\n"
    "    mad.lo.u32 %t, %i, %N, %j;\n"
    "    add.u32 %t, %t, %ro;\n"
    "    mul.wide.u32 %off, %t, 8;\n"
    "    add.u64 %off, %rdC, %off;\n"
    "    st.global.s64 [%off], %s;\n"
    "$done:\n"
    "    ret;\n"
    "}\n";

static const char *cuda_kernel_names[12] = {
    "tadd_f32", "tmul_f32", "tadd_f64", "tmul_f64",
    "tadd_i32", "tmul_i32", "tadd_i64", "tmul_i64",
    "tmm_f32", "tmm_f64", "tmm_i32", "tmm_i64"
};

// --- Backend state (single device, process lifetime) ---

static int cuda_state = 0; // 0 = unprobed, 1 = ready, -1 = unavailable
static int cuda_device_count = 0;
static CUcontext cuda_ctx = 0;
static CUmodule cuda_mod = 0;
static void *cuda_fns[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static char cuda_error[2048] = {0};
static char cuda_name[256] = {0};

static void cuda_fail(const char *msg) {
    size_t n = 0;
    while (n + 1 < sizeof(cuda_error) && msg[n]) {
        cuda_error[n] = msg[n];
        ++n;
    }
    cuda_error[n] = 0;
}

static void cuda_trace(const char *what) {
    const char *on = getenv("ALYA_TENSOR_OCL_DEBUG");
    if (on && on[0]) {
        fprintf(stderr, "[tensor-cuda] %s\n", what);
    }
}

static int cuda_load_api(void) {
    if (!cuda_lib_open()) return 0;
    CUDA_WANT(cuInit);
    CUDA_WANT(cuDeviceGetCount);
    CUDA_WANT(cuDeviceGet);
    CUDA_WANT(cuDeviceGetName);
    CUDA_WANT(cuDeviceGetAttribute);
    CUDA_WANT(cuCtxCreate_v2);
    CUDA_WANT(cuModuleLoadData);
    p_cuModuleLoadDataEx = (fn_cuModuleLoadDataEx)cuda_sym("cuModuleLoadDataEx");
    CUDA_WANT(cuModuleGetFunction);
    CUDA_WANT(cuMemAlloc_v2);
    CUDA_WANT(cuMemFree_v2);
    CUDA_WANT(cuMemcpyHtoD_v2);
    CUDA_WANT(cuMemcpyDtoH_v2);
    CUDA_WANT(cuLaunchKernel);
    CUDA_WANT(cuCtxSynchronize);
    return 1;
}

static void cuda_init(void) {
    CUdevice dev = 0;
    int ndev = 0;
    int major = 0;
    int minor = 0;
    int ki = 0;

    if (cuda_state != 0) return;
    cuda_state = -1;
    if (!cuda_load_api()) {
        cuda_fail("CUDA driver library not found");
        return;
    }
    if (p_cuInit(0) != CUDA_SUCCESS) {
        cuda_fail("cuInit failed");
        return;
    }
    if (p_cuDeviceGetCount(&ndev) != CUDA_SUCCESS || ndev <= 0) {
        cuda_fail("no CUDA device");
        return;
    }
    if (p_cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        cuda_fail("cuDeviceGet failed");
        return;
    }
    if (p_cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS
        || p_cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS
        || major < 5) {
        cuda_fail("CUDA device below sm_50");
        return;
    }
    if (p_cuDeviceGetName(cuda_name, (int)sizeof(cuda_name) - 1, dev) != CUDA_SUCCESS) {
        cuda_name[0] = 0;
    }
    if (p_cuCtxCreate_v2(&cuda_ctx, 0, dev) != CUDA_SUCCESS) {
        cuda_fail("cuCtxCreate failed");
        return;
    }
    if (p_cuModuleLoadData(&cuda_mod, (const void *)cuda_ptx_src) != CUDA_SUCCESS) {
        // Retry with an error log buffer so failures name the PTX line.
        if (p_cuModuleLoadDataEx) {
            char ptx_log[1500] = {0};
            int opt[2];
            void *optval[2];
            size_t logcap = sizeof(ptx_log) - 1;
            opt[0] = 5; // CU_JIT_ERROR_LOG_BUFFER
            optval[0] = (void *)ptx_log;
            opt[1] = 6; // CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
            optval[1] = (void *)&logcap;
            p_cuModuleLoadDataEx(&cuda_mod, (const void *)cuda_ptx_src, 2, opt, optval);
            if (ptx_log[0]) {
                size_t n = 0;
                while (n + 1 < sizeof(cuda_error) && ptx_log[n]) {
                    cuda_error[n] = ptx_log[n];
                    ++n;
                }
                cuda_error[n] = 0;
                return;
            }
        }
        cuda_fail("PTX module load failed (driver too old for PTX 6.0?)");
        return;
    }
    for (ki = 0; ki < 12; ++ki) {
        if (p_cuModuleGetFunction((CUfunction *)&cuda_fns[ki], cuda_mod, cuda_kernel_names[ki]) != CUDA_SUCCESS) {
            cuda_fns[ki] = 0;
        }
    }
    cuda_device_count = ndev;
    cuda_state = 1;
    cuda_trace("backend ready");
}

// Kernel table index by (op, dtype): op 0 = add, 1 = mul, 2 = matmul;
// dtype 0 = f64, 1 = f32, 2 = i32, 3 = i64 (all covered on CUDA).
static int cuda_kernel_index(int op, int dtype) {
    static const int table[3][4] = {
        {2, 0, 4, 6},
        {3, 1, 5, 7},
        {9, 8, 10, 11}
    };
    if (op < 0 || op > 2 || dtype < 0 || dtype > 3) return -1;
    return table[op][dtype];
}

// --- Public entry points (called from device.c) ---

int32_t alya_tensor_cuda_count(void) {
    cuda_init();
    if (cuda_state != 1) return 0;
    return (int32_t)cuda_device_count;
}

int32_t alya_tensor_cuda_supports(int32_t dtype) {
    cuda_init();
    if (cuda_state != 1) return 0;
    if (dtype >= 0 && dtype <= 3) return 1;
    return 0;
}

const char *alya_tensor_cuda_error(void) {
    return cuda_error;
}

const char *alya_tensor_cuda_name(void) {
    cuda_init();
    return cuda_name;
}

void *alya_tensor_cuda_alloc(int32_t byte_size) {
    CUdeviceptr d = 0;
    cuda_init();
    if (cuda_state != 1 || byte_size <= 0) return 0;
    if (p_cuMemAlloc_v2(&d, (size_t)byte_size) != CUDA_SUCCESS) return 0;
    return (void *)(uintptr_t)d;
}

void alya_tensor_cuda_free(void *handle) {
    if (handle) p_cuMemFree_v2((CUdeviceptr)(uintptr_t)handle);
}

int32_t alya_tensor_cuda_write(void *handle, void *host, int32_t byte_size) {
    cuda_init();
    if (cuda_state != 1 || !handle || !host || byte_size <= 0) return 0;
    if (p_cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)handle, host, (size_t)byte_size) != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

int32_t alya_tensor_cuda_read(void *handle, void *host, int32_t byte_size) {
    cuda_init();
    if (cuda_state != 1 || !handle || !host || byte_size <= 0) return 0;
    if (p_cuMemcpyDtoH_v2(host, (CUdeviceptr)(uintptr_t)handle, (size_t)byte_size) != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

static int32_t cuda_launch_ew(void *ah, void *bh, void *oh, int32_t count, int op, int32_t dtype) {
    int ki = 0;
    CUdeviceptr da = 0;
    CUdeviceptr db = 0;
    CUdeviceptr do_ = 0;
    int32_t n = 0;
    void *args[4];
    unsigned int blocks = 0;
    cuda_init();
    if (cuda_state != 1 || !ah || !bh || !oh || count <= 0) return 0;
    ki = cuda_kernel_index(op, (int)dtype);
    if (ki < 0 || !cuda_fns[ki]) return 0;
    da = (CUdeviceptr)(uintptr_t)ah;
    db = (CUdeviceptr)(uintptr_t)bh;
    do_ = (CUdeviceptr)(uintptr_t)oh;
    n = count;
    args[0] = &da;
    args[1] = &db;
    args[2] = &do_;
    args[3] = &n;
    blocks = (unsigned int)(((int64_t)count + 255) / 256);
    if (p_cuLaunchKernel((CUfunction)cuda_fns[ki], blocks, 1, 1, 256, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        return 0;
    }
    if (p_cuCtxSynchronize() != CUDA_SUCCESS) return 0;
    cuda_trace("element-wise launch ok");
    return 1;
}

int32_t alya_tensor_cuda_add(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    return cuda_launch_ew(ah, bh, oh, count, 0, dtype);
}

int32_t alya_tensor_cuda_mul(void *ah, void *bh, void *oh, int32_t count, int32_t dtype) {
    return cuda_launch_ew(ah, bh, oh, count, 1, dtype);
}

int32_t alya_tensor_cuda_matmul(void *ah, void *bh, void *oh, int32_t a_off, int32_t b_off, int32_t r_off,
                                int32_t m, int32_t n, int32_t k, int32_t dtype) {
    int ki = 0;
    CUdeviceptr da = 0;
    CUdeviceptr db = 0;
    CUdeviceptr do_ = 0;
    int32_t ao = 0;
    int32_t bo = 0;
    int32_t ro = 0;
    int32_t mm = 0;
    int32_t nn = 0;
    int32_t kk = 0;
    void *args[9];
    unsigned int gx = 0;
    unsigned int gy = 0;
    cuda_init();
    if (cuda_state != 1 || !ah || !bh || !oh || m <= 0 || n <= 0 || k <= 0) return 0;
    ki = cuda_kernel_index(2, (int)dtype);
    if (ki < 0 || !cuda_fns[ki]) return 0;
    da = (CUdeviceptr)(uintptr_t)ah;
    db = (CUdeviceptr)(uintptr_t)bh;
    do_ = (CUdeviceptr)(uintptr_t)oh;
    ao = a_off;
    bo = b_off;
    ro = r_off;
    mm = m;
    nn = n;
    kk = k;
    args[0] = &da;
    args[1] = &db;
    args[2] = &do_;
    args[3] = &ao;
    args[4] = &bo;
    args[5] = &ro;
    args[6] = &mm;
    args[7] = &nn;
    args[8] = &kk;
    gx = (unsigned int)(((int64_t)n + 15) / 16);
    gy = (unsigned int)(((int64_t)m + 15) / 16);
    if (p_cuLaunchKernel((CUfunction)cuda_fns[ki], gx, gy, 1, 16, 16, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        return 0;
    }
    if (p_cuCtxSynchronize() != CUDA_SUCCESS) return 0;
    cuda_trace("matmul launch ok");
    return 1;
}

int32_t alya_tensor_cuda_sync(void) {
    cuda_init();
    if (cuda_state != 1) return 0;
    if (p_cuCtxSynchronize() != CUDA_SUCCESS) return 0;
    return 1;
}
