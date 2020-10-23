#include "dataformats.h"
#include "StsReco.h"
#include <xpu/xpu.h>

#include <algorithm>
#include <cassert>
#include <iostream>

bool isSorted(const StsDataContainer<StsDigi> &digis) {
    bool ok = true;
    for (size_t m = 0; m < digis.nModules; m++) {
        const StsDigi *data = digis.moduleData(m);
        size_t elems = digis.size(m);
        if (elems == 0) {
            continue;
        }
        bool modOk = true;
        for (size_t i = 0; i < elems-1; i++) {
            modOk &= (data[i].time <= data[i+1].time);
        }
        ok &= modOk;
    }

    return ok;
}

void shuffleDigis(StsDataContainer<StsDigi> &digis) {
    for (size_t m = 0; m < digis.nModules; m++) {
        std::random_shuffle(digis.moduleData(m), digis.moduleData(m) + digis.size(m));
    }
}

int main() {
    xpu::initialize(xpu::driver::cpu);

    std::cout << "Loading digis..." << std::endl;
    OwningStsContainer<StsDigi> digisH = readDigis("digis_dump.txt");
    shuffleDigis(digisH);

    StsDataContainer<StsDigi> digisD{};
    digisD.set(digisH.nModules, xpu::device_malloc<size_t>(digisH.nModules + 1), xpu::device_malloc<StsDigi>(digisH.totalSize()));    

    xpu::copy(digisD.moduleOffsets, digisH.moduleOffsets, digisH.nModules + 1 );
    xpu::copy(digisD.data, digisH.data, digisH.totalSize());

    xpu::run_kernel<StsReco::sortDigis>(xpu::grid::n_blocks(digisH.nModules), digisD);

    xpu::copy(digisH.data, digisD.data, digisH.totalSize());

    assert(isSorted(digisH));

    std::cout << "Looking good!" << std::endl;

    return 0;
}