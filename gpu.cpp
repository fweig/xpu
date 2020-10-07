#include "gpu.h"

#include "GPUBackend.h"

#include <memory>

static std::unique_ptr<GPUBackend> theBackend;

namespace gpu {

    void initialize(GPUBackendType t) {
        /* theBackend = std::make_unique<CPUBackend>(); */
    }

}

