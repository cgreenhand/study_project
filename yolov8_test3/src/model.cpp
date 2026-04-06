#include "model.h"
#include "config.h"
#include <map>
#include <cassert>

static std::map<std::string, nvinfer1::Weights> loadWeight(std::string wts_path)
{

}




nvinfer1::IHostMemory* buildYolov8Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, 
                                    std::string wts_path, float gd, float gw, int max_channels)
{
    return nullptr;
}