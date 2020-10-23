#include "dataformats.h"

#include <xpu/kernel.h>
#include <algorithm>
#include <iostream>

XPU_KERNEL(sortDigis, xpu::no_smem, (StsDataContainer<StsDigi>) digis) {
    int mod = info.i_block.x;
#if XPU_CPU_CODE
    std::sort(digis.moduleData(mod), digis.moduleData(mod)+digis.size(mod), [](const StsDigi &a, const StsDigi &b) {
        return a.time < b.time;
    });
#endif
}