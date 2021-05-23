#include "this_thread.h"

thread_local xpu::dim xpu::detail::this_thread::block_idx;
thread_local xpu::dim xpu::detail::this_thread::grid_dim;
