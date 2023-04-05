# Table of Contents

- [Introduction](#introduction)
    - [Example](#example)
    - [Motivation](#motivation)
- [Integration](#integration)
    - [Requirements](#requirements)
    - [CMake](#cmake)
- [Documentation](#documentation)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)
- [Related Projects](#related-projects)

# Introduction

xpu is a tiny (< 5000 LOC) and lightweight C++ library designed to simplify GPU programming by providing a unified interface for various GPU architectures, including CPU, CUDA, HIP, and SYCL. This allows developers to write a single codebase that can be easily compiled and run on different hardware, while using modern C++ and the flexibility to use native CUDA, HIP, or SYCL code when needed.

Features include:
- Unified interface to write GPU code for CUDA, HIP, SYCL.
- Zero overhead for device code compared to native CUDA/HIP/SYCL.
- Run on CPU as fallback or for debugging
- Compile for device code for CPU with regular C++ compiler without any additional requirements
- RAII based memory management while maintaining control over how, when and where memory is allocated.
- Support for native CUDA/HIP/SYCL host code via `xpu::function` (e.g. for usage with `cub` device-wide functions).
- Common abstraction for constant memory.
- Seperate compilation of device code. Host code may call kernels from any library it's linked against.

## Example

Kernels are declared as callable objects that inherit from `xpu::kernel`. The kernel is implemented as a regular C++ function.

For example, to declare a kernel that adds two vectors in your header file:
```c++
#include <xpu/device.h>

struct DeviceLib {}; // Dummy type to match kernels to a library.

struct VectorAdd : xpu::kernel<DeviceLib> {
    using context = xpu::kernel_context<xpu::no_smem>; // optional shorthand
    XPU_D void operator()(context &,
       xpu::bufer<const float>, xpu::buffer<const float>, xpu::buffer<float>, size_t);
};
```

Then call the kernel on the host side like this:
```c++
#include <xpu/host.h>

// ...

xpu::buffer<float> a, b, c; // Declare buffers.

xpu::queue q; // Create a queue.

// Run the kernel.
q.launch<VectorAdd>(xpu::n_threads(1000), a, b, c, 1000);

// ...
```

See the [wiki](https://github.com/fweig/xpu/wiki/Vector-Add---Example) for the full runnable example.

## Motivation

I started development of `xpu` as a basis for GPU processing in the [CBM](https://www.gsi.de/work/forschung/cbmnqm/cbm) experiment.
That meant supporting as many platforms as possible, while also providing a fallback to run on CPU. Additionally i wanted to have and RAII-style memory management for device memory while retaining control of how and when allocations happen. This is something where SYCL's buffer API falls short... SYCL still also solves lot of these issues. However the problem remains the SYCL compiler could generate less performant code for our use cases and we would want to switch to a native compiler (i.e. `nvcc`) instead.

# Integration

## Requirements

`xpu` requires a C++17 capable compiler and CMake 3.11 or newer.

Additionally for GPUs a compiler for the respective backend is required:
- For Nvidia GPUs: [CUDA](https://developer.nvidia.com/cuda-toolkit) 10.2 or newer
- For AMD GPUs: [ROCm](https://www.amd.com/de/graphics/servers-solutions-rocm) 4.5 or newer
- For Intel GPUs / SYCL Targets: [Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) Compiler

Note: `xpu` can be used without a GPU backend. In this case, device code will only be compiled for CPU.

Windows is not supported at the moment. `xpu` is tested on Linux and MacOS.

## CMake

Adding `xpu` to your project is as simple as adding the following to your `CMakeLists.txt`:
```cmake
include(FetchContent)
FetchContent_Declare(xpu
    GIT_REPOSITORY https://github.com/fweig/xpu
    GIT_TAG        v0.8.0
)
FetchContent_MakeAvailable(xpu)
```

Then call `xpu_attach` on your target:
```cmake
add_library(Library SHARED ${LibrarySources}) # Works for executables as well
xpu_attach(Library ${DeviceSources}) # DeviceSources is a subset of LibrarySources that should be compiled for GPU
```

Enable the desired backends by passing `-DXPU_ENABLE_<BACKEND>=ON` to cmake. (e.g. `-DXPU_ENABLE_CUDA=ON` for CUDA or `-DXPU_ENABLE_HIP=ON` for HIP).

See the [wiki](https://github.com/fweig/xpu/wiki/CMake-Options) for all available CMake options.

# Tests

To build the tests, pass `-DXPU_ENABLE_TESTS=ON` to cmake.  To compile and run the tests, `googletest` is required.
`cmake` will automatically download and build it.

To build and run the testbench:
- `$ cmake -B build -S . -DXPU_ENABLE_TESTS=ON -DXPU_ENABLE_CUDA=ON -DXPU_ENABLE_HIP=ON -DXPU_ENABLE_SYCL=ON`
- `$ cd build`
- `$ make`
- `$ ctest .`

Disable any backends you don't need in the first step.

# Contributing

Please feel free to ask any questions you have, request features, and report bugs by creating a new [issue](https://github.com/fweig/xpu/issues/new).

# License

`xpu` is licensed under the MIT license. See [LICENSE](LICENSE) for details.

# Related projects

- Open SYCL: https://github.com/OpenSYCL/OpenSYCL
- alpaka: https://github.com/alpaka-group/alpaka
- YAKL: https://github.com/mrnorman/YAKL
- HIP-CPU: https://github.com/ROCm-Developer-Tools/HIP-CPU
- kokkos: https://github.com/kokkos/kokkos
