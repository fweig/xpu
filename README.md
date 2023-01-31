# XPU

---

## Repo Structure

- `examples/`:
    - `vector_add/`: Simple example showcasing vector addition.
    - `sorting/`: Example on how to use the sorting API.
- `src/xpu/`: source code
    - `common.h`: Public datatypes used in both host and device code.
    - `defines.h`: Public definitions.
    - `device.h`: Public device-side functions and dataypes.
    - `host.h`: Public host-side functions and datatypes.
    - `driver/`: Internal code that is platform specific (cpu, cuda or hip).
    - `detail/`: Internal code that is not tied to a specific platform.
- `templates/`: Code templates used to generate boilerplate code for device libraries.
- `test/`: Unittests

The public headers in `src/xpu` expose the entire API (With the exception of defining kernels and constant memory, see below). They are also designed to contain as little implementation details as possible, and should be therefore very easy to read. However proper documentation is missing at the moment.

---

## Configuration

CMake Options:

- `XPU_ENABLE_CUDA`: Enable / Disable compilation for cuda. (default=`OFF`)
- `XPU_CUDA_ARCH`: List of target cuda architectures. (default=`75`)
- `XPU_ENABLE_HIP`: Enable / Disable compilation for hip. (default=`OFF`)
- `XPU_HIP_ARCH`: List of target hip architectures. (default=`gfx906;gfx908`)
- `XPU_ROCM_ROOT`: Path to rocm installation. (default=`/opt/rocm`)
- `XPU_DEBUG`: Build gpu code with debug symbols and disable optimizations. (default=`OFF`)
- `XPU_BUILD_TESTS`: Build unittests and benchmarks. (default=`OFF`)
- `XPU_BUILD_EXAMPLES`: Build examples. (default=`OFF`)

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
