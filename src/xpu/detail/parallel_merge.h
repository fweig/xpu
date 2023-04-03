#ifndef XPU_DETAIL_PARALLEL_MERGE_H
#define XPU_DETAIL_PARALLEL_MERGE_H

#include "../defines.h"
#include <type_traits>

#if 0
#define PRINT_B(message, ...) if (xpu::pos.thread_idx_x() == 0) printf("t 0: " message "\n", ##__VA_ARGS__)
#define PRINT_T(message, ...) do { printf("t %d: " message "\n", xpu::pos.thread_idx_x(), ##__VA_ARGS__); } while (0)
#else
#define PRINT_B(...) ((void)0)
#define PRINT_T(...) ((void)0)
#endif

namespace xpu::detail {

// TODO: add proper support for vector types
#if !XPU_IS_CUDA_HIP
struct int4 {
    int x, y, z, w;
};
#endif

template<typename Key, int BlockSize, int ItemsPerThread>
class parallel_merge {

public:
    using data_t = Key;

    static_assert(std::is_trivially_constructible_v<data_t>, "Merged type needs trivial constructor.");

    struct storage_t {
        data_t sharedMergeMem[ItemsPerThread * BlockSize + 1];
    };

    XPU_D parallel_merge(tpos& pos, storage_t &storage) : pos(pos), storage(storage) {}

    template<typename Compare>
    XPU_D void merge(const data_t *a, size_t size_a, const data_t *b, size_t size_b, data_t *dst, Compare &&comp) {

        PRINT_B("Merging arrays of size %llu and %llu", size_a, size_b);

        int diag_next = 0;
        int mp_next = merge_path<MgpuBoundsLower>(a, size_a, b, size_b, diag_next, comp);
        for (int diag = 0; diag < static_cast<int>(size_a + size_b); diag = diag_next) {

            int mp = mp_next;

            diag_next = min(diag_next + BlockSize * ItemsPerThread, size_a + size_b);
            mp_next = merge_path<MgpuBoundsLower>(a, size_a, b, size_b, diag_next, comp);

            int4 range;
            range.x = mp;
            range.y = mp_next;
            range.z = diag - mp;
            range.w = (diag_next - mp_next);

            PRINT_B("Merging [%d, %d] and [%d, %d]", range.x, range.y, range.z, range.w);

            data_t results[ItemsPerThread];

            merge_keys(a, b, range, storage.sharedMergeMem, results, comp);

            thread_to_shared(results, storage.sharedMergeMem, true);

            // Store merged keys to global memory.
            int aCount = range.y - range.x;
            int bCount = range.w - range.z;
            shared_to_global(aCount + bCount, storage.sharedMergeMem, dst+diag, true);
        }
    }

private:
    tpos &pos;
    storage_t &storage;

    /*********************************************** KHUN BEGIN *******************************************************/

    enum MgpuBounds {
        MgpuBoundsLower,
        MgpuBoundsUpper
    };

    template<typename Compare>
    XPU_D void merge_keys(const data_t *a, const data_t *b, int4 range, data_t* data_shared, data_t* results, Compare &&comp) {

        int a0 = range.x;
        int a1 = range.y;
        int b0 = range.z;
        int b1 = range.w;

        int aCount = a1 - a0;
        int bCount = b1 - b0;

        // Load the data into shared memory.
        load_to_shared(a + a0, aCount, b + b0, bCount, data_shared, true);

        // Run a merge path to find the start of the serial merge for each thread.
        int diag = ItemsPerThread * pos.thread_idx_x();
        int diag_next = min(diag + ItemsPerThread, aCount + bCount);
        int mp = merge_path<MgpuBoundsLower>(data_shared, aCount, data_shared + aCount, bCount, diag, comp);
        int mp_next = merge_path<MgpuBoundsLower>(data_shared, aCount, data_shared + aCount, bCount, diag_next, comp);

        // Compute the ranges of the sources in shared memory.
        int a0tid = mp;
        int a1tid = mp_next;
        int b0tid = aCount + diag - mp;
        int b1tid = aCount + diag_next - mp_next;

        // PRINT_T("merge path: a0 = %d, a1 = %d, b0 = %d, b1 = %d", a0tid, a1tid, b0tid, b1tid);

        // Serial merge into register.
        serial_merge(data_shared, a0tid, a1tid, b0tid, b1tid, results, comp);
    }

    template<typename Compare>
    XPU_D void serial_merge(const data_t* keys_shared, int aBegin,  int aEnd, int bBegin, int bEnd, data_t* results, Compare &&comp) {
        data_t aKey = keys_shared[aBegin];
        data_t bKey = keys_shared[bBegin];

        constexpr bool range_check = true;

        // PRINT_T("Merging from SMEM, a0 = %d, a1 = %d, b0 = %d, b1 = %d", aBegin, aEnd, bBegin, bEnd);

        #pragma unroll
        for (int i = 0; i < ItemsPerThread && (bBegin < bEnd || aBegin < aEnd); ++i) {
            bool p;
            if (range_check) {
                p = (bBegin >= bEnd) || ((aBegin < aEnd) && !comp(bKey, aKey));
            } else {
                p = !comp(bKey, aKey);
            }

            results[i] = p ? aKey : bKey;

            // assert(aBegin < BlockSize * ItemsPerThread);
            // assert(bBegin < BlockSize * ItemsPerThread);

            // PRINT_T("aBegin = %d, bBegin = %d", aBegin, bBegin);

            if (p) {
                aKey = keys_shared[++aBegin];
            } else {
                bKey = keys_shared[++bBegin];
            }
        }
        barrier(pos);
    }

    template<MgpuBounds Bounds, typename Compare>
    XPU_D int merge_path(const data_t *a, int aCount, const data_t *b, int bCount, int diag, Compare &&comp) {

        PRINT_B("Merge path for diag %d", diag);

        int begin = max(0, diag - bCount);
        int end = min(diag, aCount);

        PRINT_B("begin = %d, end = %d", begin, end);

        while (begin < end) {
            int mid = (begin + end) >> 1;
            data_t aKey = a[mid];
            // printf("tid %d: bidx = %d\n", pos.thread_idx_x(), diag-1-mid);

            PRINT_T("b_idx = %d", diag-1-mid);
            data_t bKey = b[diag - 1 - mid];
            bool pred = (MgpuBoundsUpper == Bounds) ? comp(aKey, bKey) : !comp(bKey, aKey);

            PRINT_B("aKey = %f, bKey = %f, begin = %d, end = %d", aKey, bKey, begin, end);

            if (pred) {
                begin = mid + 1;
            } else {
                end = mid;
            }
        }
        return begin;
    }

    XPU_D void reg_to_shared(const data_t* reg, data_t *dest, bool sync) {

        #pragma unroll
        for(int i = 0; i < ItemsPerThread; ++i) {
            assert(BlockSize * i + pos.thread_idx_x() < BlockSize * ItemsPerThread);
            dest[BlockSize * i + pos.thread_idx_x()] = reg[i];
        }

        if(sync) {
            barrier(pos);
        }
    }

    XPU_D void load_to_reg(const data_t *a_global, int aCount, const data_t *b_global, int bCount, data_t *reg, bool sync)  {

        b_global -= aCount;
        int total = aCount + bCount;

        #pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            int index = BlockSize * i + pos.thread_idx_x();
            if (index < aCount) reg[i] = a_global[index];
            else if (index < total) reg[i] = b_global[index];
        }

        if (sync) {
            barrier(pos);
        }
    }

    XPU_D void load_to_shared(const data_t *a_global, int aCount, const data_t *b_global, int bCount, data_t* shared, bool sync) {
        data_t reg[ItemsPerThread];
        load_to_reg(a_global, aCount, b_global, bCount, reg, false);
        reg_to_shared(reg,  shared, sync);
    }

    XPU_D void thread_to_shared(const data_t* threadReg, data_t* shared, bool sync) {
        #pragma unroll
        for(int i = 0; i < ItemsPerThread; ++i) {
            assert(ItemsPerThread * pos.thread_idx_x() + i < BlockSize * ItemsPerThread);
            shared[ItemsPerThread * pos.thread_idx_x() + i] = threadReg[i];
        }

        if(sync) {
            barrier(pos);
        }
    }

    XPU_D void shared_to_global(int count, const data_t* source, data_t *dest, bool sync) {
        #pragma unroll
        for(int i = 0; i < ItemsPerThread; ++i) {
            int index = BlockSize * i + pos.thread_idx_x();
            if(index < count) {
                dest[index] = source[index];
            }
        }
        if(sync) {
            barrier(pos);
        }
    }

    /************************************************ KHUN END ********************************************************/
};
} // namespace xpu::detail

#endif
