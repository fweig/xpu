#pragma once

#include <xpu/defs.h>
#include <cassert>
#include <cstddef>
#include <iosfwd>
#include <unordered_map>
#include <vector>

struct StsDigi {
    float charge;
    short channel;
    short time;
};

std::ostream &operator<<(std::ostream &, const StsDigi &);

struct StsCluster {
    int size;
    float charge;
    float position;
    float positionError;
    float time;
    float timeError;
};

std::ostream &operator<<(std::ostream &, const StsCluster &);

using ModuleID = size_t;

template<typename T>
class StsDataContainer {

public:
    size_t nModules = 0;
    size_t *moduleOffsets = nullptr; // nModules + 1 entries
    T *data = nullptr;

    XPU_D bool isFront(size_t mod) const {
        return mod % 2 == 0;
    }

    XPU_D size_t totalSize() const {
        return moduleOffsets[nModules];
    }

    XPU_D size_t size(size_t mod) const {
        assert(mod < nModules);
        return moduleOffsets[mod + 1] - moduleOffsets[mod];
    }

    XPU_D T *moduleData(size_t mod) {
        assert(mod < nModules);
        return &data[moduleOffsets[mod]];
    }

    XPU_D const T *moduleData(size_t mod) const {
        assert(mod < nModules);
        return &data[moduleOffsets[mod]];
    }

    XPU_D void set(size_t nModules, size_t *moduleOffsets, T *data) {
        this->nModules = nModules;
        this->moduleOffsets = moduleOffsets;
        this->data = data;
    }

};

template<typename T>
class OwningStsContainer : public StsDataContainer<T> {
    
private:
    std::vector<size_t> moduleOffsetsO;
    std::vector<T> dataO;

public:
    OwningStsContainer(std::vector<size_t> &offset, std::vector<T> &data) : StsDataContainer<T>() {
        moduleOffsetsO = std::move(offset);
        dataO = std::move(data);
        this->set(moduleOffsetsO.size() - 1, moduleOffsetsO.data(), dataO.data());
    }

};

OwningStsContainer<StsDigi> readDigis(const std::string &);
OwningStsContainer<StsCluster> readClusters(const std::string &);