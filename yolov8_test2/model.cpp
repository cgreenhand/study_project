#include <math.h>
#include <iostream>
#include "config.h"
#include <assert.h>
#include <string>
#include "NvInfer.h"
#include "block.h"
#include "block.cpp"

static std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> WeightMap;

    std::ifstream input(file);
    assert(input.is_open() &&
           "Unable to load weight file. please check if the "
           ".wts file path is right!!!!!!");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; x++) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        WeightMap[name] = wt;
    }
    return WeightMap;
}

static int get_width(int x, float gw, int max_channels, int divisor = 8) {
    auto channel = int(ceil((x * gw) / divisor)) * divisor;
    return channel >= max_channels ? max_channels : channel;
}


static int get_depth(int x, float gd) {
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0)
        --r;
    return std::max<int>(r, 1);
}

//  计算特征图的步长
static void calculateStrides(nvinfer1::IElementWiseLayer* conv_layers[], int size, int reference_size, int strides[])
{

    for(int i = 0; i < size; i++){
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }

}

nvinfer1::IHostMemory* buildEngineYolov8Det(nvinfer1::IBuilder* builder,
                                             nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt,
                                             const std::string& wts_path,
                                             float& gd,
                                             float& gw,
                                             int& max_channels
){

// *********************************************************
//  * 1 读取权重
//  ****************************************************/

    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    // uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

  

// *********************************************************
//  * 2 YOLO8 Input
//  ****************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3,kInputH,kInputW});
    assert(data && "Failed to create input tensor.");

// *********************************************************
//  * 2 YOLO8 BACKBONE
//  ****************************************************/

    nvinfer1::IElementWiseLayer* conv0 = 
        convBnSiLU(network,weightMap,*data,get_width(64,gw,max_channels),3,2,1,"model.0");
    nvinfer1::IElementWiseLayer* conv1 = 
        convBnSiLU(network,weightMap,*conv0->getOutput(0),get_width(128,gw,max_channels),3,2,1,"model.1");
    nvinfer1::IElementWiseLayer* conv2 = 
        C2F(network,weightMap,*conv1->getOutput(0),get_width(128,gw,max_channels),get_width(128,gw,max_channels),get_depth(3,gd),true,0.5,"model.2");
    nvinfer1::IElementWiseLayer* conv3 = 
        convBnSiLU(network,weightMap,*conv2->getOutput(0),get_width(256,gw,max_channels),3,2,1,"model.3");
    nvinfer1::IElementWiseLayer* conv4 = 
        C2F(network,weightMap,*conv3->getOutput(0),get_width(256,gw,max_channels),get_width(256,gw,max_channels),get_depth(6,gd),true,0.5,"model.4");
    nvinfer1::IElementWiseLayer* conv5 = 
        convBnSiLU(network,weightMap,*conv4->getOutput(0),get_width(512,gw,max_channels),3,2,1,"model.5");
    nvinfer1::IElementWiseLayer* conv6 = 
        C2F(network,weightMap,*conv5->getOutput(0),get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(6,gd),true,0.5,"model.6");
    nvinfer1::IElementWiseLayer* conv7 = 
        convBnSiLU(network,weightMap,*conv6->getOutput(0),get_width(1024,gw,max_channels),3,2,1,"model.7");
    nvinfer1::IElementWiseLayer* conv8 = 
        C2F(network,weightMap,*conv7->getOutput(0),get_width(1024,gw,max_channels),get_width(512,gw,max_channels),get_depth(3,gd),true,0.5,"model.8");
    nvinfer1::IElementWiseLayer* conv9 = 
        SPPF(network,weightMap,*conv8->getOutput(0),get_width(1024,gw,max_channels),get_width(512,gw,max_channels),5,"model.9");

// *********************************************************
//  * 2 YOLO8 HEAD
//  ****************************************************/
    float scale[] = {1.0,2.0,2.0};
    nvinfer1::IResizeLayer* upsample10 = 
        network->addResize(*conv9->getOutput(0));
    assert(upsample10);
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale,3);

    nvinfer1::ITensor* inputTensor11[] = {upsample10->getOutput(0),conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11,2);
    
    nvinfer1::IElementWiseLayer* conv12 = 
        C2F(network,weightMap,*cat11->getOutput(0),get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(3,gd),true,0.5,"model.11");
    
    nvinfer1::IResizeLayer* upsample13 = 
        network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setScales(scale,3);

    nvinfer1::ITensor* inputTensor14[] = {upsample13->getOutput(0),conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14,2);
    
    nvinfer1::IElementWiseLayer* conv15 =
        C2F(network,weightMap,*cat14->getOutput(0),get_width(256,gw,max_channels),get_width(256,gw,max_channels),get_depth(3,gd),true,0.5,"model.15");

    nvinfer1::IElementWiseLayer* conv16 = 
        convBnSiLU(network,weightMap,*conv15->getOutput(0),get_width(256,gw,max_channels),3,2,1,"model.16");
    
    nvinfer1::ITensor* inputTensor17[]= {conv16->getOutput(0),conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17,2);

    nvinfer1::IElementWiseLayer* conv18 = 
        C2F(network,weightMap,*cat17->getOutput(0),get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(3,gd),true,0.5,"model.18");
    

    nvinfer1::IElementWiseLayer* conv19 = 
        convBnSiLU(network,weightMap,*conv18->getOutput(0),get_width(512,gw,max_channels),3,2,1,"model.19");
    
    nvinfer1::ITensor* inputTensor20[] = {conv19->getOutput(0),conv9->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20,2);
    
    nvinfer1::IElementWiseLayer* conv21 = 
        C2F(network,weightMap,*cat20->getOutput(0),get_width(1024,gw,max_channels),get_width(1024,gw,max_channels),get_depth(3,gd),true,0.5,"model.22");


// *********************************************************
//  * 2 YOLO8 OUTPUT
//  * - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
//  ****************************************************/
    int base_in_channel = (gw == 1.25) ? 80:64;
    int base_out_channle = (gw == 0.25) ? std::max(64,std::min(kNumClass,100)) : get_width(256,gw,max_channels);

    // output 0
    nvinfer1::IElementWiseLayer* conv22_cv2_0_0 = 
        convBnSiLU(network, weightMap, *conv15->getOutput(0),base_in_channel,3,1,1,"model.22.cv2.0.0");
    
    nvinfer1::IElementWiseLayer* conv22_cv2_0_1 = 
        convBnSiLU(network,weightMap,*conv22_cv2_0_0->getOutput(0),base_in_channel,3,1,1,"model.22.cv2.0.1");

    nvinfer1::IConvolutionLayer* conv22_cv2_0_2 = 
        network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0),64,nvinfer1::DimsHW{1,1},
            weightMap["model.22.cv2.0.2.weight"],weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0,0});

    nvinfer1::IElementWiseLayer* conv22_cv3_0_0 = 
        convBnSiLU(network,weightMap,*conv15->getOutput(0),base_out_channle,3,1,1,"model.22.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_0_1 = 
        convBnSiLU(network,weightMap,*conv22_cv3_0_0->getOutput(0),base_out_channle,3,1,1,"model.22.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_0_2 =
        network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0),kNumClass,nvinfer1::DimsHW{1,1},
                                weightMap["model.22.cv3.0.2.weight"],weightMap["model.22.cv3.0.2.bias"]);
    
    conv22_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv22_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});

    nvinfer1::ITensor* inputTensor22_0[] = {conv22_cv2_0_2->getOutput(0),conv22_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0,2);


    // output1 18 过来

    nvinfer1::IElementWiseLayer* conv22_cv2_1_0 =
        convBnSiLU(network,weightMap,*conv18->getOutput(0),base_in_channel,3,1,1,"model.22.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_1_1=
        convBnSiLU(network,weightMap,*conv22_cv2_1_0->getOutput(0),base_in_channel,3,1,1,"model.22.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_1_2 = 
        network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0),64,nvinfer1::DimsHW{1,1},weightMap["model.22.cv2.1.2.weight"],weightMap["model.22.cv2.1.2.bias"]);

    nvinfer1::IElementWiseLayer* conv22_cv3_1_0 =
        convBnSiLU(network,weightMap,*conv18->getOutput(0),base_out_channle,3,1,1,"model.22.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_1_1=
        convBnSiLU(network,weightMap,*conv22_cv3_1_0->getOutput(0),base_out_channle,3,1,1,"model.22.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_1_2 = 
        network->addConvolutionNd(*conv22_cv3_1_0->getOutput(0),kNumClass,nvinfer1::DimsHW{1,1},weightMap["model.22.cv3.1.2.weight"],weightMap["weight.22.cv3.1.2.bias"]);
    
    nvinfer1::ITensor* inputTensor22_1[] = {conv22_cv2_1_2->getOutput(0),conv22_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1,2);

    //output2  21 过来
    nvinfer1::IElementWiseLayer* conv22_cv2_2_0 =
        convBnSiLU(network,weightMap,*conv21->getOutput(0),base_in_channel,3,1,1,"model.22.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_2_1=
        convBnSiLU(network,weightMap,*conv22_cv2_2_0->getOutput(0),base_in_channel,3,1,1,"model.22.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_2_2 = 
        network->addConvolutionNd(*conv22_cv2_2_1->getOutput(0),64,nvinfer1::DimsHW{1,1},weightMap["model.22.cv2.2.2.weight"],weightMap["model.22.cv2.2.2.bias"]);

    nvinfer1::IElementWiseLayer* conv22_cv3_2_0 =
        convBnSiLU(network,weightMap,*conv18->getOutput(0),base_out_channle,3,1,1,"model.22.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_2_1=
        convBnSiLU(network,weightMap,*conv22_cv3_2_0->getOutput(0),base_out_channle,3,1,1,"model.22.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_2_2 = 
        network->addConvolutionNd(*conv22_cv3_2_0->getOutput(0),kNumClass,nvinfer1::DimsHW{1,1},weightMap["model.22.cv3.2.2.weight"],weightMap["weight.22.cv3.2.2.bias"]);
    
    nvinfer1::ITensor* inputTensor22_2[] = {conv22_cv2_2_2->getOutput(0),conv22_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2,2);

    /*******************************************************************************************************
  *********************************************  YOLOV8 DETECT
  *******************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers)/sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers)/sizeof(conv_layers[0]),kInputH,strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    // reshpe
    nvinfer1::IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(nvinfer1::Dims2{64+kNumClass,(kInputH / strides[0]) * (kInputH / strides[0])});
    nvinfer1::ISliceLayer* split22_0_0 = network->addSlice(
        *shuffle22_0->getOutput(0), nvinfer1::Dims2{0,0},
        nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_0_1 = network->addSlice(
        *shuffle22_0->getOutput(0),nvinfer1::Dims2{64,0},
        nvinfer1::Dims2{kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    
    nvinfer1::IShuffleLayer* dfl22_0 = 
        DFL(network,weightMap,*split22_0_0->getOutput(0),4,(kInputH / strides[0]) * (kInputW / strides[0]),1,1,0,"model.22.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor22_dfl_0[] = {dfl22_0->getOutput(0),split22_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_df_0 = network->addConcatenation(inputTensor22_dfl_0,2);
    
    

    




    return nullptr;

}


