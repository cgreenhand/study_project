#include <iostream>
#include <fstream>
#include <vector>
#include "model.h"
#include "NvInfer.h"
#include "config.h"

// 实例化 TensorRT 日志记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <weights_path> <engine_path>" << std::endl;
        std::cerr << "示例: " << argv[0] << " ./yolov8n.wts ./yolov8n.engine" << std::endl;
        return -1;
    }

    std::string wts_path = argv[1];
    std::string engine_path = argv[2];

    // 1. 初始化推理引擎构建器
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // 2. 设置模型缩放参数 (以 YOLOv8n 为例)
    // yolov8n: gd=0.33, gw=0.25
    // yolov8s: gd=0.33, gw=0.50
    // yolov8m: gd=0.67, gw=0.75
    float gd=0.33; 
    float gw=0.50;
    int max_channels = 1024;

    // 3. 调用你的构建函数
    std::cout << "开始构建网络定义..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = buildYolov8Det(
        builder, 
        config, 
        nvinfer1::DataType::kFLOAT, 
        wts_path, 
        gd, gw, max_channels
    );

    if (!serialized_model) {
        std::cerr << "错误: 构建网络失败!" << std::endl;
        return -1;
    }

    // 4. 将序列化后的引擎写入文件
    std::ofstream p(engine_path, std::ios::binary);
    if (!p) {
        std::cerr << "错误: 无法打开文件以写入引擎: " << engine_path << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
    p.close();

    std::cout << "引擎构建成功并保存至: " << engine_path << std::endl;

    // 5. 释放资源
    delete serialized_model;
    delete config;
    delete builder;

    return 0;
}