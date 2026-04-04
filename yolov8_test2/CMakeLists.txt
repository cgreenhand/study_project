cmake_minimum_required(VERSION 3.10)
project(yolov8_trt)

# 1. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 手动指定路径 (请确保这些路径在你的机器上真实存在)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(TENSORRT_DIR "/home/happy/TensorRT-8.6.1.6")

# 3. 寻找 CUDA 核心组件
find_package(CUDA REQUIRED)

# 4. 为 4060Ti (Ada Lovelace 架构) 设置 NVCC 编译选项
# sm_89 是 40系列显卡的专用架构号
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-gencode arch=compute_89,code=sm_89)

# 5. 包含头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TENSORRT_DIR}/include
)

# 6. 设置库搜索路径
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${TENSORRT_DIR}/lib
)

# 7. 编译 CUDA 插件 (yololayer.cu)
# 这会生成一个静态或动态库供主程序链接
cuda_add_library(yoloplugin SHARED yololayer.cu)

# 8. 编译主程序
# 注意：这里包含了所有必要的源文件，确保 block.cpp 不会被重复包含
add_executable(yolov8_build 
    main.cpp 
    model.cpp 
    block.cpp
)

# 9. 链接库 (采用绝对路径和标准名结合，防止路径拼接 Bug)
target_link_libraries(yolov8_build 
    yoloplugin                   # 你刚才编译的插件库
    nvinfer                      # TensorRT 核心库
    nvinfer_plugin               # TensorRT 官方插件库 (addYoloLayer 依赖)
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so  # 显式指向 CUDA 运行时
    rt                           # 系统实时库
    pthread                      # 线程库
)

# 10. (可选) 打印信息以便调试
message(STATUS "CUDA Libs: ${CUDA_LIBRARIES}")
message(STATUS "TRT Libs: ${TENSORRT_DIR}/lib")