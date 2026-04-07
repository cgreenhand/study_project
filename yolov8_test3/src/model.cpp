#include "model.h"
#include "config.h"
#include <map>
#include <cassert>
#include <math.h>
#include <fstream>
#include <iostream>
#include "block.h"
#include "yololayer.h"

static int get_width(int x, float gw, int max_channels, int divisor=8)
{
    auto channel = int(ceil((x*gw) / divisor)) * divisor;
    return channel >= max_channels? max_channels:channel;
}

// 0.5的时候 奇进偶退
static int get_depth(int x , float gd)
{
    if(x == 1) return 1;
    int r = round(x*gd);
    if(x*gd - int(x*gd) == 0.5 && (int(x*gd)%2) == 0) --r;
    return std::max<int>(r,1);
}

void static calculateStrides(nvinfer1::IElementWiseLayer* conv_layers[], int size, int reference_size, int strides[]) {
    for (int i = 0; i < size; ++i) {
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }
}
// std::map<std::string, nvinfer1::Weights> loadWeight(const std::string file) {
//     std::cout << "Loading weights: " << file << std::endl;
//     std::map<std::string, nvinfer1::Weights> WeightMap;

//     std::ifstream input(file);
//     assert(input.is_open() &&
//            "Unable to load weight file. please check if the "
//            ".wts file path is right!!!!!!");

//     int32_t count;
//     input >> count;
//     assert(count > 0 && "Invalid weight map file.");

//     while (count--) {
//         nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
//         uint32_t size;

//         std::string name;
//         input >> name >> std::dec >> size;
//         wt.type = nvinfer1::DataType::kFLOAT;

//         uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
//         for (uint32_t x = 0, y = size; x < y; x++) {
//             input >> std::hex >> val[x];
//         }
//         wt.values = val;
//         wt.count = size;
//         WeightMap[name] = wt;
//     }
//     return WeightMap;
// }

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
        input >> name >> std::dec >> size;
        // 
        if (size == 0 || name.empty()) {
            std::cout << "！！！发现异常权重项: " << name << " Size: " << size << std::endl;
        }

        wt.type = nvinfer1::DataType::kFLOAT;
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t)*size));
        for(uint32_t x = 0 ; x<size; x++){
            if (!(input >> std::hex >> val[x])) {
                std::cerr << "致命错误：权重数据读取中断！层名: " << name << " 指标: " << x << std::endl;
                exit(-1);
            }
        }
        // 
        // wt.type = nvinfer1::DataType::kFLOAT;
        // uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t)*size));
        // for(uint32_t x = 0 ; x<size; x++){
        //     input >> std::hex >> val[x];
        // }



        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;

    }
    return weightMap;

}




nvinfer1::IHostMemory* buildYolov8Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, 
                                    const std::string& wts_path, float gd, float gw, int max_channels)
{
    std::map<std::string,nvinfer1::Weights> weightMap = loadWeight(wts_path);

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

// 1. input
    nvinfer1::ITensor* data = network->addInput(yoloConfig::kInputTensorName,dt,nvinfer1::Dims3{3,yoloConfig::kinputSize,yoloConfig::kinputSize});

    assert(data && "build yolov8 det: input data error");
// 2. backbone
    nvinfer1::IElementWiseLayer* conv0 = 
            convBnSiLu(network,*data,weightMap,get_width(64,gw,max_channels),3,2,1,"model.0");
    nvinfer1::IElementWiseLayer* conv1 = 
        convBnSiLu(network,*conv0->getOutput(0),weightMap,get_width(128,gw,max_channels),3,2,1,"model.1");
    nvinfer1::IElementWiseLayer* conv2 =
        C2F(network,*conv1->getOutput(0),weightMap,get_width(128,gw,max_channels),get_width(128,gw,max_channels),get_depth(3,gd),true,0.5,"model.2");
    nvinfer1::IElementWiseLayer* conv3 = 
        convBnSiLu(network,*conv2->getOutput(0),weightMap,get_width(256,gw,max_channels),3,2,1,"model.3");
    nvinfer1::IElementWiseLayer* conv4 =
        C2F(network,*conv3->getOutput(0),weightMap,get_width(256,gw,max_channels),get_width(256,gw,max_channels),get_depth(6,gd),true,0.5,"model.4");
    nvinfer1::IElementWiseLayer* conv5 = 
        convBnSiLu(network,*conv4->getOutput(0),weightMap,get_width(256,gw,max_channels),3,2,1,"model.5");
    nvinfer1::IElementWiseLayer* conv6 =
        C2F(network,*conv5->getOutput(0),weightMap,get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(6,gd),true,0.5,"model.6");
    nvinfer1::IElementWiseLayer* conv7 = 
        convBnSiLu(network,*conv6->getOutput(0),weightMap,get_width(1024,gw,max_channels),3,2,1,"model.7");
    nvinfer1::IElementWiseLayer* conv8 =
        C2F(network,*conv7->getOutput(0),weightMap,get_width(1024,gw,max_channels),get_width(1024,gw,max_channels),get_depth(6,gd),true,0.5,"model.8");
    nvinfer1::IElementWiseLayer* conv9 =
        SPPF(network,*conv8->getOutput(0),weightMap,get_width(1024,gw,max_channels),get_width(1024,gw,max_channels),5,"model.9");

// 3. head
    float scale[] = {1.0, 2.0, 2.0}; 
    
    nvinfer1::IResizeLayer* upsample1 = network->addResize(*conv9->getOutput(0));
    upsample1->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample1->setScales(scale,3);
    
    nvinfer1::ITensor* inputTensor11[] = {upsample1->getOutput(0),conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11,2);      
    
    nvinfer1::IElementWiseLayer* conv12 = C2F(network,*cat11->getOutput(0),weightMap,get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(3,gd),false,0.5,"model.12");

    nvinfer1::IResizeLayer* upsample2 = network->addResize(*conv12->getOutput(0));
    upsample2->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample2->setScales(scale,3);

    nvinfer1::ITensor* inputTensor14[] = {upsample2->getOutput(0),conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14,2);
    // p3  256 80*80
    nvinfer1::IElementWiseLayer* conv15 = C2F(network,*cat14->getOutput(0),weightMap,get_width(256,gw,max_channels),get_width(256,gw,max_channels),get_depth(3,gd),false,0.5,"model.15");


    nvinfer1::IElementWiseLayer* conv16 = convBnSiLu(network,*conv15->getOutput(0),weightMap,256,3,2,1,"model.16");

    nvinfer1::ITensor* inputTensor17[] = {conv16->getOutput(0),conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17,2);
    //  512 p4 40*40
    nvinfer1::IElementWiseLayer* conv18 = C2F(network,*cat17->getOutput(0),weightMap,get_width(512,gw,max_channels),get_width(512,gw,max_channels),get_depth(3,gd),false,0.5,"model.18");


    nvinfer1::IElementWiseLayer* conv19 = convBnSiLu(network,*conv18->getOutput(0),weightMap,512,3,2,1,"model.19");

    nvinfer1::ITensor* inputTensor20[] = {conv19->getOutput(0),conv9->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20,2);
    // p5 1024 20*20
    nvinfer1::IElementWiseLayer* conv21 = C2F(network,*cat20->getOutput(0),weightMap,get_width(1024,gw,max_channels),get_width(1024,gw,max_channels),get_depth(3,gd),false,0.5,"model.21");

// 4. detect output
// p3 p4 p5 进行不同操作，
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(yoloConfig::kClsNum, 100)) : get_width(256, gw, max_channels);

    //  p3
    nvinfer1::IElementWiseLayer* conv22_cv2_0_0 = convBnSiLu(network,*conv15->getOutput(0),weightMap,base_in_channel,3,1,1,"model.22.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_0_1 = convBnSiLu(network,*conv22_cv2_0_0->getOutput(0),weightMap,base_in_channel,3,1,1,"model.22.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_0_2 = network->addConvolution(*conv22_cv2_0_1->getOutput(0),64,nvinfer1::DimsHW{1,1},weightMap["model.22.cv2.0.2.weight"],weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0,0});

    nvinfer1::IElementWiseLayer* conv22_cv3_0_0 = convBnSiLu(network,*conv15->getOutput(0),weightMap,base_out_channel,3,1,1,"model.22.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_0_1 = convBnSiLu(network,*conv22_cv3_0_0->getOutput(0),weightMap,base_out_channel,3,1,1,"model.22.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_0_2 = network->addConvolution(*conv22_cv3_0_1->getOutput(0),yoloConfig::kClsNum,nvinfer1::DimsHW{1,1},weightMap["model.22.cv3.0.2.weight"],weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0,0});
    nvinfer1::ITensor* inputTensor_22_0[] = {conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor_22_0,2); 

    // p4
    nvinfer1::IElementWiseLayer* conv22_cv2_1_0 = convBnSiLu(network,*conv18->getOutput(0),weightMap,base_in_channel,3,1,1,"model.22.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_1_1 = convBnSiLu(network,*conv22_cv2_1_0->getOutput(0),weightMap,base_in_channel,3,1,1,"model.22.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_1_2 = network->addConvolution(*conv22_cv2_1_1->getOutput(0),64,nvinfer1::DimsHW{1,1},weightMap["model.22.cv2.1.2.weight"],weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0,0});

    nvinfer1::IElementWiseLayer* conv22_cv3_1_0 = convBnSiLu(network,*conv18->getOutput(0),weightMap,base_out_channel,3,1,1,"model.22.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_1_1 = convBnSiLu(network,*conv22_cv3_1_0->getOutput(0),weightMap,base_out_channel,3,1,1,"model.22.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_1_2 = network->addConvolution(*conv22_cv3_1_1->getOutput(0),yoloConfig::kClsNum,nvinfer1::DimsHW{1,1},weightMap["model.22.cv3.1.2.weight"],weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0,0});
    nvinfer1::ITensor* inputTensor_22_1[] = {conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor_22_1,2); 

    // p5

    nvinfer1::IElementWiseLayer* conv22_cv2_2_0 = convBnSiLu(network,*conv21->getOutput(0),weightMap,base_in_channel,3,1,1,"model.22.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_2_1 = convBnSiLu(network,*conv22_cv2_2_0->getOutput(0),weightMap,base_in_channel,3,1,1,"model.22.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_2_2 = network->addConvolution(*conv22_cv2_2_1->getOutput(0),64,nvinfer1::DimsHW{1,1},weightMap["model.22.cv2.2.2.weight"],weightMap["model.22.cv2.2.2.bias"]);
    conv22_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0,0});

    nvinfer1::IElementWiseLayer* conv22_cv3_2_0 = convBnSiLu(network,*conv21->getOutput(0),weightMap,base_out_channel,3,1,1,"model.22.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_2_1 = convBnSiLu(network,*conv22_cv3_2_0->getOutput(0),weightMap,base_out_channel,3,1,1,"model.22.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_2_2 = network->addConvolution(*conv22_cv3_2_1->getOutput(0),yoloConfig::kClsNum,nvinfer1::DimsHW{1,1},weightMap["model.22.cv3.2.2.weight"],weightMap["model.22.cv3.2.2.bias"]);
    conv22_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0,0});
    nvinfer1::ITensor* inputTensor_22_2[] = {conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor_22_2,2); 
    

    // dfl

    // 64+cls_num, h,w
    nvinfer1::IElementWiseLayer* conv_layer[] = {conv3, conv5, conv7};
    int stridesLength = sizeof(conv_layer) / sizeof(conv_layer[0]);
    int strides[stridesLength];
    calculateStrides(conv_layer,stridesLength,yoloConfig::kinputSize,strides);
    

    // 进行dfl计算 cat22_0
    nvinfer1::IShuffleLayer* cat22_0_shuffle = network->addShuffle(*cat22_0->getOutput(0));
    cat22_0_shuffle->setReshapeDimensions(nvinfer1::Dims2{64+yoloConfig::kClsNum,(yoloConfig::kinputSize/strides[0])*(yoloConfig::kinputSize/strides[0])});
    nvinfer1::ISliceLayer* cat22_0_split_0 = 
        network->addSlice(*cat22_0_shuffle->getOutput(0),nvinfer1::Dims2{0,0},nvinfer1::Dims2{64,(yoloConfig::kinputSize / strides[0]) * (yoloConfig::kinputSize / strides[0])},nvinfer1::Dims2{1,1});
    nvinfer1::ISliceLayer* cat22_0_split_1 = 
        network->addSlice(*cat22_0_shuffle->getOutput(0),nvinfer1::Dims2{64,0},nvinfer1::Dims2{yoloConfig::kClsNum,(yoloConfig::kinputSize/strides[0])*(yoloConfig::kinputSize/strides[0])},nvinfer1::Dims2{1,1});
    // 64,grid -> 4,grid
    nvinfer1::IShuffleLayer* dfl_22_0 = 
        DFL(network,*cat22_0_split_0->getOutput(0),weightMap,4,(yoloConfig::kinputSize / strides[0]) * (yoloConfig::kinputSize / strides[0]),1,1,0,"model.22.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor_22_dfl_0[] = {dfl_22_0->getOutput(0),cat22_0_split_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* dfl_cat0 = network->addConcatenation(inputTensor_22_dfl_0,2); 
    

    // 进行dfl计算 cat22_1
    nvinfer1::IShuffleLayer* cat22_1_shuffle = network->addShuffle(*cat22_1->getOutput(0));
    cat22_1_shuffle->setReshapeDimensions(nvinfer1::Dims2{64+yoloConfig::kClsNum,(yoloConfig::kinputSize/strides[1])*(yoloConfig::kinputSize/strides[1])});
    nvinfer1::ISliceLayer* cat22_1_split_0 = 
        network->addSlice(*cat22_1_shuffle->getOutput(0),nvinfer1::Dims2{0,0},nvinfer1::Dims2{64,(yoloConfig::kinputSize / strides[1]) * (yoloConfig::kinputSize / strides[1])},nvinfer1::Dims2{1,1});
    nvinfer1::ISliceLayer* cat22_1_split_1 = 
        network->addSlice(*cat22_1_shuffle->getOutput(0),nvinfer1::Dims2{64,0},nvinfer1::Dims2{yoloConfig::kClsNum,(yoloConfig::kinputSize/strides[1])*(yoloConfig::kinputSize/strides[1])},nvinfer1::Dims2{1,1});
    // 64,grid -> 4,grid
    nvinfer1::IShuffleLayer* dfl_22_1 = 
        DFL(network,*cat22_1_split_0->getOutput(0),weightMap,4,(yoloConfig::kinputSize / strides[1]) * (yoloConfig::kinputSize / strides[1]),1,1,0,"model.22.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor_22_dfl_1[] = {dfl_22_1->getOutput(0),cat22_1_split_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* dfl_cat1 = network->addConcatenation(inputTensor_22_dfl_1,2); 
   
    // 进行dfl计算 cat22_2
    nvinfer1::IShuffleLayer* cat22_2_shuffle = network->addShuffle(*cat22_2->getOutput(0));
    cat22_2_shuffle->setReshapeDimensions(nvinfer1::Dims2{64+yoloConfig::kClsNum,(yoloConfig::kinputSize/strides[2])*(yoloConfig::kinputSize/strides[2])});
    nvinfer1::ISliceLayer* cat22_2_split_0 = 
        network->addSlice(*cat22_2_shuffle->getOutput(0),nvinfer1::Dims2{0,0},nvinfer1::Dims2{64,(yoloConfig::kinputSize / strides[2]) * (yoloConfig::kinputSize / strides[2])},nvinfer1::Dims2{1,1});
    nvinfer1::ISliceLayer* cat22_2_split_1 = 
        network->addSlice(*cat22_2_shuffle->getOutput(0),nvinfer1::Dims2{64,0},nvinfer1::Dims2{yoloConfig::kClsNum,(yoloConfig::kinputSize/strides[2])*(yoloConfig::kinputSize/strides[2])},nvinfer1::Dims2{1,1});
    // 64,grid -> 4,grid
    nvinfer1::IShuffleLayer* dfl_22_2 = 
        DFL(network,*cat22_2_split_0->getOutput(0),weightMap,4,(yoloConfig::kinputSize / strides[2]) * (yoloConfig::kinputSize / strides[2]),1,1,0,"model.22.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor_22_dfl_2[] = {dfl_22_2->getOutput(0),cat22_2_split_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* dfl_cat2 = network->addConcatenation(inputTensor_22_dfl_2,2); 

// 5. yoloLayer 处理->  p3: (4+cls_num,6400)  p4: (4+cls_num,1600)  p5: (4+cls_num,400)
//   4表示是在代表在特征图上的 ltrb   经过yoloLayer变为 在kInputSize上的 ltrb 
   
    // 此处按住不表
    std::vector<nvinfer1::IConcatenationLayer *> dets = {dfl_cat0,dfl_cat1,dfl_cat2};
    nvinfer1::IPluginV2Layer* yolo =
        addYoLoLayer(network,dets,strides,stridesLength,yoloConfig::kClsNum,false,false,false);

    yolo->getOutput(0)->setName(yoloConfig::kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(yoloConfig::kBatchSize);
    config->setMaxWorkspaceSize(1*(1<<30));
#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, kInputQuantizationFolder, "int8calib.table",
                                                  kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network,*config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for(auto& mem: weightMap){
        free((void*)(mem.second.values));
    }

    return serialized_model;

}   