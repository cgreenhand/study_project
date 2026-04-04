#include "NvInfer.h"
#include <iostream>
#include <fstream>
#include <memory>

// --- 第一步：定义一个简单的日志类 ---
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印警告和错误
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger; // 实例化一个全局对象 gLogger

// 声明你的函数
extern nvinfer1::IHostMemory* buildEngineYolov8Det(nvinfer1::IBuilder* builder, 
    nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, 
    const std::string& wts_path, float& gd, float& gw, int& max_channels);

int main() {
    // --- 第二步：使用刚才定义的 gLogger ---
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        std::cerr << "Failed to create builder" << std::endl;
        return -1;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    
    // 根据你的模型修改参数 (yolov8s 示例)
    float gd = 0.33f; 
    float gw = 0.50f;
    int max_channels = 1024;

    std::cout << "Starting to build engine..." << std::endl;
    auto plan = buildEngineYolov8Det(builder.get(), config.get(), 
                                    nvinfer1::DataType::kFLOAT, "yolov8s.wts", 
                                    gd, gw, max_channels);
    
    if (plan) {
        std::ofstream p("yolov8.engine", std::ios::binary);
        p.write(reinterpret_cast<const char*>(plan->data()), plan->size());
        std::cout << "Engine build successfully saved to yolov8.engine!" << std::endl;
    } else {
        std::cerr << "Failed to build engine plan!" << std::endl;
    }

    return 0;
}