# XPU

## Introduction

`xpu` is a tiny C++ library  that provides a unified interface to run kernels on CPU, CUDA, HIP and SYCL. It is designed to be used in conjunction with a device library that is compiled for CPU, CUDA and HIP. The device library is created by compiling a subset of the sources as device code. The device library is then linked against the host code. The host code can then call kernels on CPU or GPU.

---

## Requirements

- C++17 capable compiler
- CMake 3.11 or newer
- For Nvidia GPUs: CUDA 10.2 or newer (optional)
- For AMD GPUs: ROCm 4.0 or newer (optional)
- For Intel GPUs / SYCL Targets: Intel oneAPI DPC++ Compiler (optional)

Note: `xpu` doesn't support Windows at the moment.

---

## Getting started

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

---

## Example

*TODO: Move this to the wiki. Add a shorter version here.*

Kernels are declared as callable objects that inherit from `xpu::kernel`. The kernel is implemented as a regular C++ function. The function header must be wrapped with the `XPU_KERNEL` macro. The kernel is then exported by calling `XPU_EXPORT_KERNEL` or `XPU_EXPORT`. The kernel can then be called on the host side by calling `xpu::run_kernel`.

For example, declare and implement a kernel that adds two vectors in your header file:
```c++
#include <xpu/device.h>

struct DeviceLib {}; // Dummy type to match kernels to a library.

struct VectorAdd : xpu::kernel<DeviceLib> {
    using context = xpu::kernel_context<xpu::no_smem>; // optional shorthand
    XPU_D void operator()(context &, const float *, const float *, float *, size_t);
};
```

Then implement the kernel in a source file:
```c++
#include "VectorAdd.h"

XPU_IMAGE(DeviceLib); // Define the device library. This call must happen in exactly one source file.

XPU_EXPORT(VectorAdd); // Export the kernel.
XPU_D void VectorAdd::operator()(context &ctx, const float *a, const float *b, float *c, size_t n) {
    size_t i = ctx.block_idx_x() * ctx.block_dim_x() + ctx.thread_idx_x(); // Get the global thread index.
    if (i >= n) return; // Check if we are out of bounds.
    c[i] = a[i] + b[i];
}
```

Finally, call the kernel on the host side:
```c++
#include <xpu/host.h>
#include "VectorAdd.h"

int main() {
    xpu::initialize(); // Initialize xpu. Must be called before any other xpu function.

    // Create buffers on the host and device.
    xpu::buffer<float> a(1000, xpu::io_buffer);
    xpu::buffer<float> b(1000, xpu::io_buffer);
    xpu::buffer<float> c(1000, xpu::io_buffer);

    xpu::h_view a_view = a.view(); // Access buffer data on the host.
    xpu::h_view b_view = b.view();

    // Fill buffers with data.
    for (size_t i = 0; i < a_view.size(); ++i)
        a_view[i] = b_view[i] = i;

    // Transfer data to the device.
    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);

    // Run the kernel.
    xpu::run_kernel<VectorAdd>(xpu::n_threads(a_view.size()), a, b, c, a_view.size());

    // Transfer data back to the host.
    xpu::copy(c, xpu::device_to_host);

    // Check the result.
    xpu::h_view c_view = c.view();
    for (size_t i = 0; i < c_view.size(); ++i)
        assert(c_view[i] == 2 * i);
}
```

---

## Configuration

CMake Options:



Environment variables:

- `XPU_PROFILE`: Enable collecting kernel times. (default=`0`)
- `XPU_VERBOSE`: Print debug information about memory allocations / memory transfer / kernel runs. (default=`0`)
- `XPU_DEVICE`: Select the device to run kernels on. Values must have the form "`<driver><devicenr>`".  If `devicenr` is missing, defaults to device 0 of selected driver. Possible values are for example: `cpu`, `cuda0`, `cuda1`, `hip0`. (default=`cpu`)

---

## Tests and examples

Building unittests requires the `googletest` framework. `xpu` will download and build `googletest` automatically.

Setup a build folder to compile both unittests and the vector add example:
```
mkdir build && cd build
cmake -DXPU_BUILD_TESTS=ON -DXPU_BUILD_EXAMPLES=ON ..
```
To enable compilation for cuda, pass `-DXPU_ENABLE_CUDA=ON` to cmake as well. (Or call `cmake -DXPU_ENABLE_CUDA=ON .` after the first cmake call.)

To build and run the tests:
```
make
XPU_DEVICE=cpu LD_LIBRARY_PATH=.:test ./test/xpu_test
```
If cuda was enabled, use `XPU_DEVICE=cuda0` to run the tests on the first Nvidia GPU instead.

Run the examples with:
```
cd build

# Run vector add
LD_LIBRARY_PATH=.:examples/vector_add ./examples/vector_add/vector_add

# Run sorting
LD_LIBRARY_PATH=.:examples/sorting ./examples/sorting/sort_struct
```

---

## Defining and running kernels

A device library is a regular shared library, where some of the sources are declared as device code. These device sources may then be compiled again as GPU code for CUDA or HIP. During runtime the application can choose to run kernels on CPU (always available) or on GPU (if available).

### Creating a device library

In addition to the device sources, a special def-file is required that describes the available kernels in order to create a device library. The def-file is described below.

Assuming you have those things, creating the device library is done in CMake by calling `xpu_attach`:
```
add_library(Library SHARED ${LibrarySources})
xpu_attach(Library ${DeviceSources})
```
`DeviceSources` should the subset of `LibrarySources` that is compiled for GPU plus any additional dependencies that may be required. The files in `DeviceSources` are compiled for GPU as a unity-build without linking to against any additional libraries.

### Declaring and implementing kernels

Kernels are declared inside header-files as Functors that inherit from xpu::kernel.
For example:
```
#include <xpu/device.h>

struct DeviceLib {}; // Dummy type to match kernels to a library.

XPU_BLOCK_SIZE(VectorAdd, 128); // Set the block size. Default is 64.

XPU_EXPORT_KERNERL(DeviceLib, VectorAdd, const float *, const float *, float *, size_t);

struct VectorAdd : xpu::kernel<DeviceLib> {
    using block_size = xpu::block_size<128>; // Optional: Set block size, default is 64.
    using shared_memory = SMemType; // Declare shared memory type,
                                    // defaults to xpu::no_smem to that no shared memory should be allocated.
    using context = xpu::kernel_context<shared_memory>; // optional shorthand
    XPU_D void operator()(context &ctx, const float *, const float *, float *, size_t);
};

```
would declare a kernel `VectorAdd` that receives three float-pointers and an integer of type `size_t` as arguments.

A kernel is implemented like any regular C++-function. But the function header must be wrapped with the `XPU_KERNEL` macro. The syntax is as follows:
```c++
#include "DeviceLib.h"

XPU_IMAGE(DeviceLib); // XPU_IMAGE must be called exactly once in one of device
                      // sources

XPU_EXPORT(VectorAdd); // Export kernel. Only then is available to be called via xpu::run_kernel in host code.

// Kernel implementation
XPU_D void VectorAdd::operator()(context &ctx, const float *a, const float *b, float *c, size_t n) {
    // Your code here
}
```

### Calling a kernel on the host side

To call a kernel from the host side, you need to include the device-library header, and
`xpu/host.h`. After initializing `xpu`, kernels may be started by calling `xpu::run_kernel`. The kernel is passed as a template argument. Along with the kernel arguments, an instance of `xpu::grid` must be passed to describe the dimensions of the kernel.

Example:
```c++
#include "deviceLib.h"
#include <xpu/host.h>

int main() {
    xpu::initialize();

    xpu::run_kernel<kernelName>(
        xpu::grid::n_blocks(NumOfBlocks), arg0, arg1, ...);

    return 0;
}
```

These calls can either happen inside the host-code of your device library. Or any code that is linked against your device library and `xpu`.

---

## Using constant memory

TODO

---

## Related projects

- Open SYCL: https://github.com/OpenSYCL/OpenSYCL
- alpaka: https://github.com/alpaka-group/alpaka
- YAKL: https://github.com/mrnorman/YAKL
- HIP-CPU: https://github.com/ROCm-Developer-Tools/HIP-CPU
- kokkos: https://github.com/kokkos/kokkos
