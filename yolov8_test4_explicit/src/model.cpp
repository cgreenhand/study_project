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

nvinfer1::IHostMemory* buildYolov8Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig, nvinfer1::DataType dt, const std::string& wts_path, float gd, float gw, int max_channels)
{
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeight(wts_path);
    
    auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

    // input
    nvinfer1::ITensor* data = network->addInput(yoloConfig::kInputTensorName, dt, nvinfer1::Dims4{-1,3,yoloConfig::kInputSize,yoloConfig::kInputSize});

    // backbone
    nvinfer1::IElementWiseLayer* conv0 = 
        convBnSiLu(network,*data,weightMap,get_width(64,gw,max_channels),3,2,1,"model.0");
    nvinfer1::IElementWiseLayer* conv1 =
        convBnSiLu(network,*conv0->getOutput(0),weightMap,get_width(64,gw,max_channels),3,2,1,"model.1");
    nvinfer1::IElementWiseLayer* conv2 = 
        C2F(network,*conv1->getOutput(0),weightMap,get_width(128,gw,max_channels),get_width(128,gw,max_channels),get_depth(3,gd),true,0.5,"model.2");
    nvinfer1::IElementWiseLayer* conv3 = 
        convBnSiLu(network,*conv2->getOutput(0),weightMap,get_width(256,gw,max_channels),3,2,1,"model.3");
    nvinfer1::IElementWiseLayer* conv4 = 
        C2F(network,*conv3->getOutput(0),weightMap,get_width(256,gw,max_channels),get_width(256,gw,max_channels),get_depth(6,gd),true,0.5,"model.4");
    nvinfer1::IElementWiseLayer* conv5 = 
        convBnSiLu(network,*conv4->getOutput(0),weightMap,get_width(512,gw,max_channels),3,2,1,"model.5");
    nvinfer1::IElementWiseLayer* conv6 = 
        C2F(network,*conv5->getOutput(0),weightMap,get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(6,gd),true,0.5,"model.6");
    nvinfer1::IElementWiseLayer* conv7 = 
        convBnSiLu(network,*conv7->getOutput(0),weightMap,get_width(512,gw,max_channels),3,2,1,"model.7");
    nvinfer1::IElementWiseLayer* conv8 = 
        C2F(network,*conv7->getOutput(0),weightMap,get_width(1024,gw,max_channels),get_width(1024,gw,max_channels),get_depth(3,gd),true,0.5,"model.8");
    nvinfer1::IElementWiseLayer* conv9 =
        SPPF(network,*conv8->getOutput(0),weightMap,get_width(1024,gw,max_channels),get_width(1024,gw,max_channels),5,"model.9");

    
// head
    float scale[] = {1.0,1.0, 2.0, 2.0}; 
    nvinfer1::IResizeLayer* upsample10 =
        network->addResize(*conv9->getOutput(0));
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale,4);
    
    nvinfer1::ITensor* inputTensor11[] = {upsample10->getOutput(0),conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11,2);
    cat11->setAxis(1);
    //  TODO 这里通道是否正确呢
    nvinfer1::IElementWiseLayer* conv12 = 
        C2F(network,*cat11->getOutput(0),weightMap,get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(3,gd),false,0.5,"model.12");
    
    // =============== //
    nvinfer1::IResizeLayer* upsample13 =
        network->addResize(*conv12->getOutput(0));
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setScales(scale,4);
    
    nvinfer1::ITensor* inputTensor14[] = {upsample13->getOutput(0),conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14,2);
    cat14->setAxis(1);
    //  TODO 这里通道是否正确呢
    nvinfer1::IElementWiseLayer* conv15 = 
        C2F(network,*cat14->getOutput(0),weightMap,get_width(256,gw,max_channels),get_width(256,gw,max_channels),get_depth(3,gd),false,0.5,"model.15");
    
    // =============== //
    nvinfer1::IElementWiseLayer* conv16 = 
        convBnSiLu(network,*conv15->getOutput(0),weightMap,get_width(256,gw,max_channels),3,2,1,"model.16");
    
    nvinfer1::ITensor* inputTensor17[] = {conv16->getOutput(0),conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17,2);
    cat17->setAxis(1);
    //  TODO 这里通道是否正确呢
    nvinfer1::IElementWiseLayer* conv18 = 
        C2F(network,*cat17->getOutput(0),weightMap,get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(3,gd),false,0.5,"model.18");
   
    // =============== //
    nvinfer1::IElementWiseLayer* conv19 = 
        convBnSiLu(network,*conv18->getOutput(0),weightMap,get_width(512,gw,max_channels),3,2,1,"model.19");
    
    nvinfer1::ITensor* inputTensor20[] = {conv19->getOutput(0),conv9->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20,2);
    cat17->setAxis(1);
    //  TODO 这里通道是否正确呢
    nvinfer1::IElementWiseLayer* conv21 = 
        C2F(network,*cat20->getOutput(0),weightMap,get_width(1024,gw,max_channels),get_width(1024,gw,max_channels),get_depth(3,gd),false,0.5,"model.21");
    
    // Detect 三个检测头 每个检测头分 box和cls  box走DFL cls正常结果
    // p3 p4 p5 进行不同操作




}