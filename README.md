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
- `XPU_ROCM_ROOT`: Path to rocm. (default=`/opt/rocm`)
- `XPU_DEBUG`: Build gpu code with debug symbols and disable optimizations. (default=`OFF`)
- `XPU_BUILD_TESTS`: Build unittests and benchmarks. (default=`OFF`)
- `XPU_BUILD_EXAMPLES`: Build examples. (default=`OFF`)

Environment variables:

- `XPU_PROFILE`: Enable collecting kernel times. (default=`0`)
- `XPU_DEVICE`: Select the device to run kernels on. Values must have the form "`<driver><devicenr>`".  If `devicenr` is missing, defaults to device 0 of selected driver. Possible values are for example: `cpu`, `cuda0`, `cuda1`, `hip0`. (default=Device selected when calling `xpu::initialize`)

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
XPU_DRIVER=cpu LD_LIBRARY_PATH=.:test ./test/xpu_test
```
If cuda was enabled, use `XPU_DRIVER=cuda` to run the tests on a GPU instead.

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
add_library(Library SHARED LibrarySources)
xpu_attach(Library DeviceSources)
```
`DeviceSources` should the subset of `LibrarySources` that is compiled for GPU plus any additional dependencies that may be required. The files in `DeviceSources` are compiled for GPU as a unity-build without linking to against any additional libraries.

### Declaring and implementing kernels

Kernels are declared inside header-files like regular C++ functions, but the declaration is wrapped inside the `XPU_EXPORT_KERNEL` macro.
For example:
```
#include <xpu/device.h>

struct DeviceLib {}; // Dummy type to match kernels to a library.

XPU_BLOCK_SIZE(VectorAdd, 128); // Set the block size. Default is 64.

XPU_EXPORT_KERNERL(DeviceLib, VectorAdd, const float *, const float *, float *, size_t);
```
would declare a kernel `VectorAdd` that receives three float-pointers and an integer of type `size_t` as arguments.

A kernel is implemented like any regular C++-function. But the function header must be wrapped with the `XPU_KERNEL` macro. The syntax is as follows:
```c++
#include "DeviceLib.h"

XPU_IMAGE(DeviceLib); // XPU_IMAGE must be called exactly once in one of device
                      // sources

XPU_KERNEL(kernelName, SMemType, Type0 arg0, Type1 arg1, ...) {
    // Your code here
}
```
Where `deviceLib` is the name of the device library that contains the kernel,
`kernelName` is the name of the kernel, `SMemTyp` is the type that is allocated in shared memory.
(The special type `xpu::no_smem` may be used to indicate that a kernel doesn't need to allocate shared memory.)
Followed by the arguments that the kernel receives.

### Calling a kernel on the host side

To call a kernel from the host side, you need to include the device-library header, and
`xpu/host.h`. After initializing `xpu`, kernels may be started by calling `xpu::run_kernel`. The kernel is passed as a template argument. Along with the kernel arguments, an instance of `xpu::grid` must be passed to describe the dimensions of the kernel.

Example:
```c++
#include "deviceLib.h"
#include <xpu/host.h>

int main() {
    xpu::initialize(xpu::driver::cpu); // or xpu::driver::cuda

    xpu::run_kernel<kernelName>(
        xpu::grid::n_blocks(NumOfBlocks), arg0, arg1, ...);

    return 0;
}
```

These calls can either happen inside the host-code of your device library. Or any code that is linked against your device library and `xpu`.

---

## Using constant memory

TODO
