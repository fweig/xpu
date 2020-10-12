#define XPU_DEVICE_LIBRARY_BACKEND_NAME VectorOpsCPU
#define XPU_DEVICE_LIBRARY_FRONTEND_H "../../../../examples/vector_add/VectorOps.h"

#include <xpu/device_library/template/backend.h>

// TODO include singleton...
#include <xpu/device_library/template/frontend.cpp>

#include "vec_add.cpp"