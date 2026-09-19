# tensor

[![CI](https://github.com/alya-lang/tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/alya-lang/tensor/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/alya-lang/tensor?color=blue&label=License)](LICENSE)
[![Alya](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Falya-lang%2Ftensor%2Fmain%2Falya.toml&query=%24.package.alya-version&label=Alya&color=orange&prefix=%3E%3D)](https://github.com/alya-lang/alya)
[![Package Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Falya-lang%2Ftensor%2Fmain%2Falya.toml&query=%24.package.version&label=Version&color=brightgreen)](alya.toml)

Multi-dimensional tensor engine with SIMD vector acceleration

---

## 🌟 Features

- ⚡ **Lightweight & Fast**: Built for speed with minimal overhead
- 🧩 **Modular Architecture**: Multi-module design with public API facade (`src/lib.alya`), data models (`src/types.alya`), and subfolder hierarchies (`src/core/formatter.alya`)
- 🔒 **Public/Private Visibility (`pub`)**: Explicit export control with `pub` for public functions, structs, and enums, keeping internal helper functions private and encapsulated
- 🎨 **Modern Syntax & Features**: First-class `enum` variants, struct methods (`config.summary()`), pattern matching with `when`, and null-checks (`is null`)
- 🛡️ **Reliable & Typed**: Explicit struct definitions, default parameters, and clean namespaced APIs
- 🧪 **Well Tested**: Comprehensive test suite with standard assertions

---

## 📁 Project Architecture

```
tensor/
├── alya.toml               # Package manifest
├── c/                      # (Optional) Native C sources for zero-dependency FFI packages
├── src/
│   ├── lib.alya            # Public API facade (pub exports & private sanitizers)
│   ├── types.alya          # Data models, pub enum, pub struct, and struct methods
│   ├── ffi.alya            # (Optional) Native extern "C" declarations
│   └── core/               # Subdirectory module hierarchy (optional for larger packages)
│       └── formatter.alya  # Domain formatting logic, pub helpers & private when matchers
├── examples/
│   └── demo.alya           # Runnable usage examples
├── tests/
│   └── test_basic.alya     # Automated test suite
└── benches/
    └── bench_basic.alya    # Micro-benchmarks
```

> [!NOTE]
> **Visibility & Modularity:** Symbols annotated with `pub` (`pub function`, `pub struct`, `pub enum`) are exported to callers and re-exporting modules. Symbols without `pub` remain strictly private/internal to their declaring module, preventing naming collisions and accidental symbol leakage.

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
import "tensor" as pkg

function main()
    # 1. Basic facade call with default parameter
    let greeting = pkg::hello()
    say f"Greeting:  {greeting}"

    # 2. Struct configuration with enum style and struct method
    let cfg = pkg::new_config("Community", 5, pkg::TensorStyle.Formal)
    say f"Summary:   {cfg.summary()}"
    say f"Formatted: {pkg::core_format_custom(cfg)}"
end

main()
```

---

## 📖 API Reference

| Symbol | Visibility | Description |
|---|---|---|
| `hello(name = "World")` | `pub function` | Returns a friendly greeting message. Defaults to `"World"` if null or omitted. |
| `new_config(name, count, style)` | `pub function` | Constructs a new configuration struct with defaults (`"World"`, `1`, `Standard`). |
| `TensorStyle` | `pub enum` | Enumeration of available greeting styles (`Standard`, `Formal`, `Casual`). |
| `TensorConfig` | `pub struct` | Configuration data model (`name`, `prefix`, `count`, `style`). |
| `TensorConfig.summary()` | `pub method` | Returns formatted string summary representation. |
| `TensorConfig.with_name(new_name)` | `pub method` | Returns an updated configuration copy with a new validated name. |
| `core_format_greeting(name)` | `pub function` | Core formatter producing `Hello, {name}!`. |
| `core_format_custom(config)` | `pub function` | Formats greeting using prefix, style (via `when`), and name from config. |

> [!TIP]
> **Internal Helpers & Documentation:** Public symbols are documented with `##` Markdown docstrings, enabling automatic API documentation generation via `alya doc`. Private functions such as `internal_clean_name` in `src/lib.alya` and `build_salutation` in `src/core/formatter.alya` are not annotated with `pub` and remain encapsulated within their respective modules.

---

## 🧪 Running Tests, Benchmarks & Documentation

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
   ```
5. Commit your changes (`git commit -m "feat: add feature"`) and open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.