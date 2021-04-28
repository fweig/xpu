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

## Tests and examples

Building unittests requires the `googletest` framework. `xpu` will download and build `googletest` automatically.

Setup a build folder to compile both unittests and the vector add example:
```
mkdir build && cd build
cmake -DXPU_ENABLE_TESTS=ON -DXPU_ENABLE_EXAMPLES=ON ..
```
To enable compilation for cuda, pass `-DXPU_ENABLE_CUDA=ON` to cmake as well. (Or call `cmake -DXPU_ENABLE_CUDA=ON .` after the first cmake call.)

To build and run the tests:
```
make
XPU_TEST_DRIVER=cpu LD_LIBRARY_PATH=.:test ./test/xpu_test
```
If cuda was enabled, use `XPU_TEST_DRIVER=cuda` to run the tests on a GPU instead.

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

Assuming you have those things, creating the device library is done in CMake by calling `xpu_attach_device_libary`:
```
add_library(Library SHARED LibrarySources)
xpu_attach_device_library(Library DeviceLibrary DefFile DeviceSources)
```
This call would result in the creation of a header `DeviceLibrary.h` using the def-file `DefFile` where all available kernels are defined. Kernels are available on the host-side as `DeviceLibrary::KernelName`.
`DeviceSources` should the subset of `LibrarySources` that is compiled for GPU plus any additional dependencies that may be required. The files in `DeviceSources` are compiled for GPU as a unity-build without linking to against any additional libraries.

### Declaring and implementing kernels

Kernels are declared inside a special def-file. Kernels are declared like C++ functions, but the declaration is wrapped inside the macro `XPU_KERNEL_DECL`.
For example:
```
XPU_KERNEL_DECL(vectorAdd, const float *, const float *, float *, size_t)
```
would declare a kernel `vectorAdd` that receives three float-pointers and an integer of type `size_t` as arguments.

The contents of the def-file are just a list of these kernel declarations.

For kernels that depend on custom datatypes, headers may be included at the top of the def-file enclosed by an `#ifdef XPU_INCLUDE` guard. E.g.:
```c++
#ifdef XPU_INCLUDE
#include <some_type.h>
class another_type; // forward declarations are also ok
#endif

XPU_KERNEL_DECL(kernelName, some_type *, another_type *)
```

A kernel is implemented like any regular C++-function. But the function header must be wrapped with the `XPU_KERNEL` macro. The syntax is as follows:
```c++
#include "deviceLib.h"

XPU_KERNEL(deviceLib, kernelName, SMemType, (Type0) arg0, (Type1) arg1, ...) {
    // Your code here
}
```
Where `deviceLib` is the name of the device library that contains the kernel,
`kernelName` is the name of the kernel, `SMemTyp` is the type that is allocated in shared memory.
(The special type `xpu::no_smem` may be used to indicate that a kernel doesn't need to allocate shared memory.)
Followed by the arguments that the kernel receives. Note that the brackets around the argument types are necessary.

### Calling a kernel on the host side

To call a kernel from the host side, you need to include the device-library header, and
`xpu/host.h`. After initializing `xpu`, kernels may be started by calling `xpu::run_kernel`. The kernel is passed as a template argument. Along with the kernel arguments, an instance of `xpu::grid` must be passed to describe the dimensions of the kernel.

Example:
```c++
#include "deviceLib.h"
#include <xpu/host.h>

int main() {
    xpu::initialize(xpu::driver::cpu); // or xpu::driver::cuda

    xpu::run_kernel<deviceLib::kernelName>(
        xpu::grid::n_blocks(NumOfBlocks), arg0, arg1, ...);

    return 0;
}
```

These calls can either happen inside the host-code of your device library. Or any code that is linked against your device library and `xpu`.

---

## Using constant memory

TODO
