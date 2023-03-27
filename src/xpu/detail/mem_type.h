#ifndef XPU_DETAIL_MEM_TYPE_H
#define XPU_DETAIL_MEM_TYPE_H

namespace xpu::detail {

enum mem_type {
    host,
    device,
    shared,
    unknown,
};

};

#endif // XPU_DETAIL_MEM_TYPE_H
