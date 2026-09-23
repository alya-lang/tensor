# tensor

[![CI](https://github.com/alya-lang/tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/alya-lang/tensor/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/alya-lang/tensor?color=blue&label=License)](LICENSE)
[![Alya](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Falya-lang%2Ftensor%2Fmain%2Falya.toml&query=%24.package.alya-version&label=Alya&color=orange&prefix=%3E%3D)](https://github.com/alya-lang/alya)
[![Package Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Falya-lang%2Ftensor%2Fmain%2Falya.toml&query=%24.package.version&label=Version&color=brightgreen)](alya.toml)

High-performance N-dimensional Tensor engine with hardware-accelerated SIMD GEMM matrix multiplication and vector operations for Alya.

---

## 🌟 Features

- ⚡ **SIMD Accelerated GEMM**: 4-wide unrolled matrix multiplication with FMA3/Neon hardware intrinsics
- 💾 **Real Dtype Storage**: `Float64` (default), `Float32`, `Int32`, and `Int64` buffers with native-width loads/stores (`std/mem` narrow accessors over `cvtss2sd`/`movslq` hardware conversion)
- 🧮 **Exact Integer Arithmetic**: integer dtypes compute in integer arithmetic (no float mediation); same-dtype enforcement with loud `throw` instead of silent promotion
- ➕ **Element-Wise Vector Ops**: Vectorized addition (`add`), subtraction (`sub`), Hadamard multiplication (`mul`), division (`div`), and scaling (`scale`) — all with strict shape checking
- 📊 **Fast Reductions**: Horizontal reduction summing (`sum`) plus `mean`, `min`, and `max` statistics
- 🔄 **Zero-Copy Reshaping**: Stride-aware multidimensional views (`reshape`, `flatten`, `to_array`, `to_string` with 2D matrix rendering)
- 🔁 **Shape Algebra**: Contiguous `transpose`, deep `copy`, `full`/`eye` constructors, in-place `fill`, and `allclose` tolerance comparison
- 📡 **NumPy-Style Broadcasting**: Implicit right-aligned stretching in `add`/`sub`/`mul`/`div` (size-1 dimensions stretch, incompatible shapes throw), explicit zero-copy `broadcast_to` views, and `broadcast_shape` shape algebra
- 🎛️ **Device Placement (Phase 0)**: Per-tensor `device_id` tagging (`CPU`/`SIMD`/`GPU`), explicit `to()` transfer staging, `synchronize()` barrier, and a native accelerator registry (`c/device.c`) — no backend registered yet, so compute honestly falls back to CPU/SIMD (see GPU roadmap below)
- 🛡️ **Loud Shape Errors**: Element-wise mismatches and bad reshapes `throw` instead of silently producing garbage
- 🔒 **Public/Private Visibility (`pub`)**: Strict encapsulation of memory internals and buffer pointers
- 🧪 **Thoroughly Tested & Benchmarked**: Comprehensive unit test suite (301 assertions) and micro-benchmarks
- 🧮 **Element-Wise Math**: `neg`, `abs`, `sqrt`, `exp`, `ln`, `pow`, `clip` (exact integer paths where closed; direct native calls, immune to inference hazards)
- 📉 **Extended Reductions**: `prod`, population `variance`/`std`, `argmin`/`argmax`
- 📦 **Batched GEMM**: Rank-3+ `matmul` with broadcast batch dimensions (strided, view-consistent)

---

## 📁 Project Architecture

```
tensor/
├── alya.toml               # Package manifest
├── src/
│   ├── lib.alya            # Public API facade (Tensor struct, SIMD GEMM, element ops)
│   ├── types.alya          # TensorDtype, TensorDevice, TensorConfig data models
│   └── core/
│       ├── formatter.alya  # Vector & matrix string representation logic
│       └── device.alya     # Device placement API + native registry FFI (Phase 0)
├── c/
│   └── device.c            # Accelerator registry (no backend registered yet)
├── examples/
│   └── demo.alya           # Runnable showcase (GEMM, dtypes, broadcast, algebra, devices)
├── tests/
│   └── test_basic.alya     # Automated test suite (301 assertions)
└── benches/
    └── bench_basic.alya    # Micro-benchmarks (allocation, GEMM, reductions)
```

> [!NOTE]
> **Visibility & Modularity:** Symbols annotated with `pub` (`pub struct Tensor`, `pub enum TensorDevice`, `pub function Tensor.matmul`) are exported to callers. Memory alignment and raw pointer management remain safely encapsulated within the package.

---

## 📦 Installation

Add `tensor` to the `[dependencies]` section in your `alya.toml`:

```toml
[dependencies]
tensor = { git = "https://github.com/alya-lang/tensor", branch = "main" }
```

Or install it directly using the Alya package CLI:

```bash
alya add tensor --git https://github.com/alya-lang/tensor --branch main
alya install
```

---

## 🚀 Quick Start

```alya
import "tensor" as tensor

function main()
    # 1. Create tensors from arrays
    let A = tensor::Tensor.from_array([2, 3], [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    ])

    let B = tensor::Tensor.from_array([3, 2], [
        7.0, 8.0,
        9.0, 1.0,
        2.0, 3.0
    ])

    # 2. Perform SIMD GEMM matrix multiplication (2x3 * 3x2 -> 2x2)
    let C = A.matmul(B)
    say "Result C(0,0): " + str(C.get_2d(0, 0)) # 31.0
    say "Result C(1,1): " + str(C.get_2d(1, 1)) # 55.0

    # 3. Element-wise operations & reductions
    let scaled = C.scale(2.0)
    say "Total Sum: " + str(scaled.sum())

    # 4. Clean up allocated heap memory
    A.free()
    B.free()
    C.free()
    scaled.free()
end

main()
```

---

## 📖 API Reference

| Symbol | Visibility | Description |
|---|---|---|
| `Tensor.new(shape, dtype)` | `pub function` | Allocates uninitialized Tensor of given shape with 32-byte cache-line alignment (`dtype`: `0` = Float64, `1` = Float32, `2` = Int32, `3` = Int64). |
| `Tensor.zeros(shape, dtype)` | `pub function` | Creates a new Tensor initialized with all `0` (converts to storage dtype). |
| `Tensor.ones(shape, dtype)` | `pub function` | Creates a new Tensor initialized with all `1` (converts to storage dtype). |
| `Tensor.from_array(shape, arr, dtype)` | `pub function` | Creates a new Tensor from a flat **float** element array (`[1.0, 2.0]`, not `[1, 2]`). |
| `Tensor.from_array_int(shape, arr, dtype)` | `pub function` | Creates a new Tensor from a flat **integer** element array (exact storage, incl. Int64). |
| `Tensor.arange(n, dtype)` | `pub function` | Creates rank-1 `[0..n)` (integer-exact). |
| `Tensor.arange_step(start, stop, step, dtype)` | `pub function` | Creates rank-1 integer-stepped `[start..stop)` (throws on zero step). |
| `Tensor.linspace(start, stop, num, dtype)` | `pub function` | Creates rank-1 with `num` evenly spaced points (inclusive). |
| `Tensor.randn(shape, dtype)` | `pub function` | Creates a Tensor with standard-normal random values (Box-Muller). |
| `Tensor.full(shape, val, dtype)` | `pub function` | Creates a new Tensor with every element set to `val`. |
| `Tensor.zeros_like(ref)` | `pub function` | Zero tensor matching reference shape/dtype/device. |
| `Tensor.ones_like(ref)` | `pub function` | Ones tensor matching reference shape/dtype/device. |
| `Tensor.full_like(ref, val)` | `pub function` | Filled tensor matching reference shape/dtype/device. |
| `Tensor.diag(self)` | `pub method` | Builds an n-by-n diagonal matrix from a rank-1 tensor. |
| `Tensor.diagonal(self)` | `pub method` | Extracts the main diagonal as a rank-1 copy. |
| `Tensor.triu(self, k)` | `pub method` | Upper triangle (NumPy `k` offset semantics). |
| `Tensor.tril(self, k)` | `pub method` | Lower triangle (NumPy `k` offset semantics). |
| `Tensor.eye(n, dtype)` | `pub function` | Creates an n-by-n identity matrix. |
| `Tensor.copy(self)` | `pub method` | Deep copy with a freshly allocated buffer (no aliasing). |
| `Tensor.to_dtype(self, dtype)` | `pub method` | Converts storage dtype (float-mediated; 2^53 caveat for Int64). |
| `Tensor.fill(self, val)` | `pub method` | Overwrites every element in place. |
| `Tensor.get_flat_int(self, idx)` | `pub method` | Exact integer element access (no float mediation). |
| `Tensor.set_flat_int(self, idx, val)` | `pub method` | Integer element update (integer storage). |
| `Tensor.get_int(self, indices)` | `pub method` | Exact integer N-d element access. |
| `Tensor.set_int(self, indices, val)` | `pub method` | Integer N-d element update. |
| `Tensor.matmul(self, other)` | `pub method` | SIMD-accelerated 2D GEMM ($M \times K \times N$); rank-3+ runs batched GEMM with broadcast batch dims. |
| `Tensor.add(self, other)` | `pub method` | Vectorized element-wise addition with broadcasting. |
| `Tensor.add(self, other)` | `pub method` | Vectorized element-wise addition of two matching-shape tensors. |
| `Tensor.sub(self, other)` | `pub method` | Vectorized element-wise subtraction with broadcasting. |
| `Tensor.mul(self, other)` | `pub method` | Vectorized element-wise multiplication (Hadamard product) with broadcasting. |
| `Tensor.div(self, other)` | `pub method` | Vectorized element-wise division with broadcasting (IEEE-754 semantics on divide-by-zero). |
| `Tensor.scale(self, factor)` | `pub method` | Multiplies all elements by a scalar float multiplier. |
| `Tensor.neg(self)` | `pub method` | Element-wise negation (exact per dtype). |
| `Tensor.abs(self)` | `pub method` | Element-wise absolute value (exact per dtype). |
| `Tensor.sqrt(self)` | `pub method` | Element-wise square root (int storage truncates on store). |
| `Tensor.exp(self)` | `pub method` | Element-wise natural exponential (int storage truncates on store). |
| `Tensor.ln(self)` | `pub method` | Element-wise natural logarithm (int storage truncates on store). |
| `Tensor.pow(self, exponent)` | `pub method` | Element-wise power; exact repeated squaring for non-negative integral exponents on int storage. |
| `Tensor.clip(self, lo, hi)` | `pub method` | Element-wise clamp into `[lo, hi]` (exact per dtype). |
| `Tensor.sum(self)` | `pub method` | Computes horizontal scalar sum of all elements. |
| `Tensor.prod(self)` | `pub method` | Computes the scalar product of all elements (empty product is `1`). |
| `Tensor.mean(self)` | `pub method` | Computes the arithmetic mean of all elements. |
| `Tensor.variance(self)` | `pub method` | Computes the population variance (divides by N). |
| `Tensor.std(self)` | `pub method` | Computes the population standard deviation. |
| `Tensor.min(self)` | `pub method` | Finds the minimum element value. |
| `Tensor.max(self)` | `pub method` | Finds the maximum element value. |
| `Tensor.argmin(self)` | `pub method` | Returns the flat index of the minimum element (first occurrence wins). |
| `Tensor.argmax(self)` | `pub method` | Returns the flat index of the maximum element (first occurrence wins). |
| `Tensor.argsort(self)` | `pub method` | Ascending sort positions as an Int32 index tensor (stable mergesort). |
| `Tensor.sort(self)` | `pub method` | All elements sorted ascending as a rank-1 tensor. |
| `Tensor.topk(self, k)` | `pub method` | `k` largest values, descending (pair with `argsort` for positions). |
| `Tensor.cumsum(self)` | `pub method` | Flat-order prefix sums with the input shape. |
| `Tensor.get_2d(self, row, col)` | `pub method` | Fast 2D matrix element access. |
| `Tensor.set_2d(self, row, col, val)`| `pub method` | Fast 2D matrix element mutation. |
| `Tensor.reshape(self, new_shape)` | `pub method` | Creates a reshaped view with updated dimension strides. Throws on element-count mismatch. |
| `Tensor.squeeze(self)` | `pub method` | View with size-1 dimensions removed. |
| `Tensor.unsqueeze(self, dim)` | `pub method` | View with a size-1 dimension inserted at `dim`. |
| `Tensor.permute(self, axes)` | `pub method` | Contiguous copy with reordered dimensions (validates the permutation). |
| `Tensor.slice(self, dim, start, stop)` | `pub method` | Contiguous copy sliced along one dimension (strict bounds). |
| `Tensor.concat(tensors, dim)` | `pub function` | Concatenates same-rank, same-dtype tensors along `dim`. |
| `Tensor.stack(tensors, dim)` | `pub function` | Stacks same-shaped tensors along a new dimension. |
| `Tensor.split(self, parts, dim)` | `pub method` | Splits into `parts` equal contiguous chunks (even division required). |
| `Tensor.broadcast_to(self, shape)` | `pub method` | Zero-copy broadcast view stretched to `shape` (stride-0 dims, shared buffer). |
| `broadcast_shape(a, b)` | `pub function` | Computes the right-aligned broadcast output shape of two shape arrays. |
| `Tensor.flatten(self)` | `pub method` | Returns a flattened rank-1 (`[size]`) view sharing the same buffer. |
| `Tensor.transpose(self)` | `pub method` | Returns the 2D transpose as a new contiguous tensor. |
| `Tensor.allclose(self, other, tol)` | `pub method` | Approximate element-wise equality within absolute tolerance `tol` (default `1e-9`); `false` on shape mismatch. |
| `Tensor.to_array(self)` | `pub method` | Converts all tensor elements to a standard flat Alya float array. |
| `Tensor.to_array_int(self)` | `pub method` | Converts all tensor elements to a flat Alya integer array (exact for integer storage). |
| `Tensor.save(self, path)` | `pub method` | Serializes to a text file (bit-exact `ALYA-TENSOR-1` format). |
| `Tensor.load(path)` | `pub function` | Deserializes a `save` file (shape, dtype, device, bits). |
| `Tensor.free(self)` | `pub method` | Releases 32-byte aligned buffer from heap memory. |
| `TensorDtype` | `pub enum` | Supported numerical data types (`Float64`, `Float32`, `Int64`, `Int32`). |
| `TensorDevice` | `pub enum` | Target execution device backend (`CPU`, `SIMD`, `GPU`). |
| `TensorConfig` | `pub struct` | Execution profile and threading configuration model. |
| `Tensor.to(self, device)` | `pub method` | Exact staged copy placed on another device (shares nothing with the source). |
| `Tensor.device_of(self)` | `pub method` | Returns the placement tag (`0` = CPU, `1` = SIMD, `2` = GPU). |
| `Tensor.is_accelerated(self)` | `pub method` | `true` only for GPU-placed tensors on machines with a registered backend. |
| `device_has_accelerator()` | `pub function` | Queries the native registry; `false` until a platform backend registers. |
| `synchronize()` | `pub function` | Ordering barrier for device streams (documented no-op while execution is synchronous). |

> [!NOTE]
> **Storage model:** buffers hold native-width elements (`Float64` default, `Float32`, `Int32`, `Int64` via `std/mem` narrow accessors). There is no implicit cross-dtype promotion: element-wise kernels require identical dtypes and `throw` on mismatch — convert explicitly with `to_dtype`. Value parameters are explicitly `float`, so pass `2.0`, not `2` (whole-program inference compiles each call site by its static type); integer element arrays go through `from_array_int` / `set_flat_int`. `get_flat` converts integer storage to float (exact below 2^53); use `get_flat_int` beyond that. Element-wise shape mismatches and bad reshapes `throw` instead of silently producing garbage.

---

## 🎛️ Device Execution & GPU Roadmap

Phase 0 (shipped): the placement API is fully wired — `device_id` tagging, `to()` staging copies, `synchronize()` barrier, and the native registry (`c/device.c`, linked via `[build] c-sources`). The registry reports **0 accelerators on every OS**, so compute runs the CPU/SIMD kernels. That fallback is deliberate and tested (`is_accelerated() == false`), never a silent fake GPU — code written against `to()` today runs unchanged when backends land.

| Step | Status | Notes |
|---|---|---|
| Placement API (`to`, `device_of`, `synchronize`) | ✅ Shipped | Tested (§14), 146 assertions green |
| Native registry + FFI path (`c/device.c`) | ✅ Shipped | Compiles/links on all OSes via `[build]`; returns honest 0 |
| CPU-fallback numerics for device-tagged tensors | ✅ Shipped | Placement preserved through ops, values exact |
| Metal backend (macOS) | ⬜ Open | Needs display-machine + `c/device_metal.m` behind `c-sources-macos` |
| CUDA backend (Windows/Linux) | ⬜ Open | Needs CUDA toolkit + `c/device_cuda.c` behind per-OS sources |
| OpenCL fallback backend | ⬜ Open | Portable fallback once the registry protocol grows kernel entries |
| Async streams (`synchronize` becomes a real fence) | ⬜ Open | Needs `sy::spawn` worker-thread queue in `device.alya` |

---

## 🧪 Running Tests & Benchmarks

Run the automated test suite using `alya test`:

```bash
alya test
```

Generate static API documentation:

```bash
alya doc . -o docs --markdown
```

Run the benchmark suite:

```bash
alya run benches/bench_basic.alya
```

Run the example demo:

```bash
alya run examples/demo.alya
```

Check code formatting:

```bash
alya fmt . --check
```

Run static code linter:

```bash
alya lint . --check
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository and clone it locally
2. Install dependencies:
   ```bash
   alya install
   ```
3. Create your feature branch (`git checkout -b feature/my-feature`)
4. Verify tests and formatting before opening a PR:
   ```bash
   alya test
   alya run benches/bench_basic.alya
   ```
5. Commit your changes (`git commit -m "feat: add feature"`) and open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.