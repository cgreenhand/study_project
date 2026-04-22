#include "model.h"
#include "config.h"
#include <map>
#include <cassert>
#include <math.h>
#include <fstream>
#include <iostream>
#include "block.h"
#include "yoloplugin.h"

static int get_width(int x, float gw, int max_channels, int divisor=8)
{
    auto channel = int(ceil(x*gw/divisor)) * divisor;
    return channel >= max_channels ? max_channels : channel;
}

// 0.5 的时候 奇进偶退
static int get_depth(int x, float gd)
{
    if(x == 1) return 1;
    int r = round(x * gd);
    if((x*gd - r) == 0.5 && x%2==0){
        r = r-1;
    }

    return std::max<int>(r,1);
}


void static calculateStrides(nvinfer1::IElementWiseLayer* conv_layer[], int size, int reference_size, int strides[])
{
    for(int i = 0 ; i<size; ++i){
        nvinfer1::ILayer* layer = conv_layer[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }

}

static std::map<std::string, nvinfer1::Weights> loadWeight(const std::string& wts_path)
{
    std::cout << "Loading weights: " << wts_path << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap = {};

    std::ifstream input(wts_path);
    assert(input.is_open() &&
           "Unable to load weight file. please check if the "
           ".wts file path is right!!!!!!");  
           
    int32_t count;
    input >> count;
    while(count --){
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        std::string name;
        input >> name >> std::dec >>size;

        if(size == 0 || name.empty()){
           std::cout << "！！！发现异常权重项: " << name << " Size: " << size << std::endl; 
        }

        wt.type = nvinfer1::DataType::kFLOAT;
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t)*size));
        for(uint32_t x =0; x<size; x++){
            input >> std::hex >> val[x];
        }

        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}


