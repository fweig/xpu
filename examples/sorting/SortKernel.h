#include "KeyValuePair.h"
#include <xpu/device.h>
#include <cstddef> // for size_t

struct SortKernel {};

XPU_EXPORT_KERNEL(SortKernel, GpuSort, KeyValuePair *, KeyValuePair *, KeyValuePair **, size_t);
