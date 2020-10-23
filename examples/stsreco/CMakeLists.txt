add_executable( run_sts_reco
    run_sts_reco.cpp
    dataformats.cpp
)

xpu_add_device_library(StsReco KERNEL_DEF kernels.def INTERFACE_H dataformats.h SRCS StsReco.cpp)
target_link_libraries(run_sts_reco xpu StsReco)