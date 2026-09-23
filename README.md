# tensor

[![CI](https://github.com/alya-lang/tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/alya-lang/tensor/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/alya-lang/tensor?color=blue&label=License)](LICENSE)
[![Alya](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Falya-lang%2Ftensor%2Fmain%2Falya.toml&query=%24.package.alya-version&label=Alya&color=orange&prefix=%3E%3D)](https://github.com/alya-lang/alya)
[![Package Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Falya-lang%2Ftensor%2Fmain%2Falya.toml&query=%24.package.version&label=Version&color=brightgreen)](alya.toml)

High-performance N-dimensional Tensor engine with hardware-accelerated SIMD GEMM matrix multiplication and vector operations for Alya.

---

## 🌟 Features

- ⚡ **SIMD Accelerated GEMM**: 4-wide unrolled matrix multiplication with FMA3/Neon hardware intrinsics
- 🧠 **Cache-Aligned Memory**: 32-byte cache-line aligned raw memory buffers (`std/mem` aligned_alloc)
- ➕ **Element-Wise Vector Ops**: Vectorized addition (`add`), subtraction (`sub`), Hadamard multiplication (`mul`), division (`div`), and scaling (`scale`) — all with strict shape checking
- 📊 **Fast Reductions**: Horizontal reduction summing (`sum`) plus `mean`, `min`, and `max` statistics
- 🔄 **Zero-Copy Reshaping**: Stride-aware multidimensional views (`reshape`, `flatten`, `to_array`, `to_string` with 2D matrix rendering)
- 🔁 **Shape Algebra**: Contiguous `transpose`, deep `copy`, `full`/`eye` constructors, in-place `fill`, and `allclose` tolerance comparison
- 🛡️ **Loud Shape Errors**: Element-wise mismatches and bad reshapes `throw` instead of silently producing garbage
- 🔒 **Public/Private Visibility (`pub`)**: Strict encapsulation of memory internals and buffer pointers
- 🧪 **Thoroughly Tested & Benchmarked**: Comprehensive unit test suite (82 assertions) and micro-benchmarks

---

## 📁 Project Architecture

```
tensor/
├── alya.toml               # Package manifest
├── src/
│   ├── lib.alya            # Public API facade (Tensor struct, SIMD GEMM, element ops)
│   ├── types.alya          # TensorDtype, TensorDevice, TensorConfig data models
│   └── core/
│       └── formatter.alya  # Vector & matrix string representation logic
├── examples/
│   └── demo.alya           # Runnable usage examples
├── tests/
│   └── test_basic.alya     # Automated test suite (82 assertions)
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
| `Tensor.new(shape)` | `pub function` | Allocates uninitialized Tensor of given shape with 32-byte cache-line alignment. |
| `Tensor.zeros(shape)` | `pub function` | Creates a new Tensor initialized with all `0.0`. |
| `Tensor.ones(shape)` | `pub function` | Creates a new Tensor initialized with all `1.0`. |
| `Tensor.from_array(shape, arr)` | `pub function` | Creates a new Tensor initialized from a flat float array. |
| `Tensor.full(shape, val)` | `pub function` | Creates a new Tensor with every element set to `val`. |
| `Tensor.eye(n)` | `pub function` | Creates an n-by-n identity matrix. |
| `Tensor.copy(self)` | `pub method` | Deep copy with a freshly allocated buffer (no aliasing). |
| `Tensor.fill(self, val)` | `pub method` | Overwrites every element in place. |
| `Tensor.matmul(self, other)` | `pub method` | SIMD-accelerated 2D GEMM matrix multiplication ($M \times K \times N$). |
| `Tensor.add(self, other)` | `pub method` | Vectorized element-wise addition of two matching-shape tensors. |
| `Tensor.sub(self, other)` | `pub method` | Vectorized element-wise subtraction of two matching-shape tensors. |
| `Tensor.mul(self, other)` | `pub method` | Vectorized element-wise multiplication (Hadamard product). |
| `Tensor.div(self, other)` | `pub method` | Vectorized element-wise division (IEEE-754 semantics on divide-by-zero). |
| `Tensor.scale(self, factor)` | `pub method` | Multiplies all elements by a scalar float multiplier. |
| `Tensor.sum(self)` | `pub method` | Computes horizontal scalar sum of all elements. |
| `Tensor.mean(self)` | `pub method` | Computes the arithmetic mean of all elements. |
| `Tensor.min(self)` | `pub method` | Finds the minimum element value. |
| `Tensor.max(self)` | `pub method` | Finds the maximum element value. |
| `Tensor.get_2d(self, row, col)` | `pub method` | Fast 2D matrix element access. |
| `Tensor.set_2d(self, row, col, val)`| `pub method` | Fast 2D matrix element mutation. |
| `Tensor.reshape(self, new_shape)` | `pub method` | Creates a reshaped view with updated dimension strides. Throws on element-count mismatch. |
| `Tensor.flatten(self)` | `pub method` | Returns a flattened rank-1 (`[size]`) view sharing the same buffer. |
| `Tensor.transpose(self)` | `pub method` | Returns the 2D transpose as a new contiguous tensor. |
| `Tensor.allclose(self, other, tol)` | `pub method` | Approximate element-wise equality within absolute tolerance `tol` (default `1e-9`); `false` on shape mismatch. |
| `Tensor.to_array(self)` | `pub method` | Converts all tensor elements to a standard flat Alya float array. |
| `Tensor.free(self)` | `pub method` | Releases 32-byte aligned buffer from heap memory. |
| `TensorDtype` | `pub enum` | Supported numerical data types (`Float64`, `Float32`, `Int64`, `Int32`). |
| `TensorDevice` | `pub enum` | Target execution device backend (`CPU`, `SIMD`). |
| `TensorConfig` | `pub struct` | Execution profile and threading configuration model. |

> [!NOTE]
> **Storage model:** tensor buffers always hold 64-bit floats (`Float64`). `TensorDtype`, `TensorDevice`, and `TensorConfig` are descriptive metadata for execution profiles — no alternate-dtype storage or GPU backend exists yet. Element-wise kernels require exactly matching shapes and `throw` on mismatch; `reshape` throws (instead of returning the input) when the element count differs.

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