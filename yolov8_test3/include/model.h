#pragma once
#include <string>
#include "NvInfer.h"





nvinfer1::IHostMemory* buildYolov8Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, 
                                    std::string wts_path, float gd, float gw, int max_channels);
