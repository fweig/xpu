#ifndef XPU_DETAIL_MACROS_H
#define XPU_DETAIL_MACROS_H

#define XPU_CONCAT_I(a, b) a##b
#define XPU_CONCAT(a, b) XPU_CONCAT_I(a, b)

#define XPU_MAGIC_NAME(prefix) XPU_CONCAT(prefix, __LINE__)

#define XPU_MAYBE_UNUSED [[gnu::unused]]

#endif
