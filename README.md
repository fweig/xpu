# Xpu

## Tests and examples

Building unittests requires the `googletest` framework. (E.g. install under Ubuntu with `sudo apt install googletest`)

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

Run the example with:
```
LD_LIBRARY_PATH=.:examples/vector_add ./examples/vector_add/vector_add
```
