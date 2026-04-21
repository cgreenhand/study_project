#include "block.h"
#include <math.h>
#include <cassert>
#include "config.h"
#include <iostream>
#include <vector>


// 实现batchnorm 和 silu
static nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                        std::map<std::string, nvinfer1::Weights>& weightMap, std::string lname, float eps)
{

    const float* running_mean = static_cast<const float*>(weightMap[lname+".running_mean"].values);
    const float* running_var = static_cast<const float*>(weightMap[lname+".running_var"].values);
    const float* gamma = static_cast<const float*>(weightMap[lname+".weight"].values);
    const float* beta = static_cast<const float*>(weightMap[lname+".bias"].values);
    const int len = weightMap[lname+".running_mean"].count;

    float* scval = reinterpret_cast<float *>(malloc(sizeof(float)*len));
    for(int i = 0;i<len;i++){
        scval[i] = gamma[i] / sqrtf(running_var[i] + eps); 
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT,scval,len};

    // TODO 是否修改
    float* shval = reinterpret_cast<float *>(malloc(sizeof(float)* len));
    for(int i=0;i<len;i++){
        shval[i] = beta[i] - scval[i] * running_mean[i];
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT,shval,len};

    float* powval = reinterpret_cast<float *>(malloc(sizeof(float)* len));
    for(int i=0;i<len;i++){
        powval[i] = 1.0f;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT,powval,len};

   

    weightMap[lname+ ".shift"] =shift;
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* output = network->addScale(input,nvinfer1::ScaleMode::kCHANNEL,shift,scale,power);
    if (!output) {
        std::cerr << "addScale 失败，层名: " << lname << std::endl;
        exit(-1);
    }
    return output;

}

static inline nvinfer1::IElementWiseLayer* addSiLU(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input)
{
    nvinfer1::IActivationLayer* sigmoid = network->addActivation(input,nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* output = network->addElementWise(*sigmoid->getOutput(0),input,nvinfer1::ElementWiseOperation::kPROD);

    assert(output && "siLU error");
    return output;
}


nvinfer1::IElementWiseLayer* convBnSiLu(nvinfer1::INetworkDefinition* network, 
                                        nvinfer1::ITensor& input, std::map<std::string, nvinfer1::Weights>& weightMap,
                                        int ch, int k, int s, int p, std::string lname)
{
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT,nullptr,0};
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(input,ch,nvinfer1::DimsHW{k,k},weightMap[lname+".conv.weight"],bias);
    conv1->setStrideNd(nvinfer1::DimsHW{s,s});
    conv1->setPaddingNd(nvinfer1::DimsHW{p,p});
    assert(conv1 && "convBnSiLu -> conv1 error");

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network,*conv1->getOutput(0),weightMap,lname + ".bn",1e-3);
    assert(bn && "convBnSiLu -> bn error");
    
    nvinfer1::IElementWiseLayer* silu = addSiLU(network,*bn->getOutput(0));
    assert(silu && "convBnSiLu -> siLu error");

    return silu;


}


static nvinfer1::IElementWiseLayer* addBottleNeck(nvinfer1::INetworkDefinition* network,nvinfer1::ITensor& input,
                                                    std::map<std::string, nvinfer1::Weights>& weightMap, int ch1, int ch2 , bool shortcut, float e, std::string lname)
{

    // int c_ = int (ch2 * e);

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLu(network,input,weightMap,ch1,3,1,1,lname+".cv1");
    assert(conv1 && "addBottleNeck cv1 error");

    nvinfer1::IElementWiseLayer* conv2 = convBnSiLu(network,*conv1->getOutput(0),weightMap,ch2,3,1,1,lname+".cv2");
    assert(conv2 && "addBottleNeck cv2 error");
    
    if(shortcut && ch1 == ch2){
        nvinfer1::IElementWiseLayer* add1 = network->addElementWise(input,*conv2->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
        assert(add1 && "addBottleNeck add1 error");
        return add1;
    }

    return conv2;

}




nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weigthMap, int c_in, int c_out, int nd,
                                bool shortcut, float e, std::string lname)
{
    int c_ = int((float)c_out * e);
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLu(network,input,weigthMap,c_out,1,1,0,lname+".cv1");

    nvinfer1::Dims dims = conv1->getOutput(0)->getDimensions();

    nvinfer1::ISliceLayer* split1 = network->addSlice(*conv1->getOutput(0),nvinfer1::Dims4{0,0,0,0},nvinfer1::Dims4{dims.d[0],dims.d[1]/2,dims.d[2],dims.d[3]},nvinfer1::Dims4{1,1,1,1});
    nvinfer1::ISliceLayer* split2 = network->addSlice(*conv1->getOutput(0),nvinfer1::Dims4{0,dims.d[1]/2,0,0},nvinfer1::Dims4{dims.d[0],dims.d[1]/2,dims.d[2],dims.d[3]},nvinfer1::Dims4{1,1,1,1});

    std::vector<nvinfer1::ITensor*> inputTensors;
    inputTensors.reserve(nd+2);
    inputTensors.push_back(split1->getOutput(0));
    inputTensors.push_back(split2->getOutput(0));

    nvinfer1::ITensor* temp_tensor = split2->getOutput(0);

    for(int i = 0; i < nd; i++){
        nvinfer1::IElementWiseLayer* bot = addBottleNeck(network,*temp_tensor,weigthMap,c_,c_,true,0.5,lname+".m."+std::to_string(i));
        temp_tensor = bot->getOutput(0);
        inputTensors.push_back(temp_tensor);
    }

    nvinfer1::IConcatenationLayer* cat1 = network->addConcatenation(inputTensors.data(),inputTensors.size());
    cat1->setAxis(1);
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLu(network,*cat1->getOutput(0),weigthMap,c_out,1,1,0,lname+".cv2");

    return conv2;

}


nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weightMap,int c_in, int c_out, int k, std::string lname)
{
    int c_ = int(c_out/2);

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLu(network,input,weightMap,c_,1,1,0,lname+".cv1");
    nvinfer1::IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0), nvinfer1::PoolingType::kMAX,nvinfer1::DimsHW{k,k});
    pool1->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool1->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});

    nvinfer1::IPoolingLayer* pool2 = network->addPoolingNd(*pool1->getOutput(0), nvinfer1::PoolingType::kMAX,nvinfer1::DimsHW{k,k});
    pool2->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool2->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});

    nvinfer1::IPoolingLayer* pool3 = network->addPoolingNd(*pool2->getOutput(0), nvinfer1::PoolingType::kMAX,nvinfer1::DimsHW{k,k});
    pool3->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool3->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});

    std::vector<nvinfer1::ITensor*> concatTensors = {
        conv1->getOutput(0), 
        pool1->getOutput(0), 
        pool2->getOutput(0), 
        pool3->getOutput(0)
    };

    nvinfer1::IConcatenationLayer* cat1 = network->addConcatenation(concatTensors.data(),concatTensors.size());
    cat1->setAxis(1);

    nvinfer1::IElementWiseLayer* conv2  = convBnSiLu(network,*cat1->getOutput(0),weightMap,c_out,1,1,0,lname+".cv2");

    return conv2;
}

//  box_out -> bot_out2
//  (n,4*16,grid) -> (n,4,grid)
nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weightMap,
                                int ch, int grid, int k , int s, int p, std::string lname)
{   
    nvinfer1::IShuffleLayer* shuff1 = network->addShuffle(input);
    shuff1->setReshapeDimensions(nvinfer1::Dims4{0,4,16,grid});
    shuff1->setSecondTranspose(nvinfer1::Permutation{0,2,1,3});
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*shuff1->getOutput(0));
    softmax->setAxes(1 << 1);

    nvinfer1::Weights bias = {nvinfer1::DataType::kFLOAT, nullptr,0};
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*softmax->getOutput(0),1,nvinfer1::DimsHW{1,1},weightMap[lname],bias);
    conv1->setStrideNd(nvinfer1::DimsHW{s,s});
    conv1->setPaddingNd(nvinfer1::DimsHW{p,p});

    nvinfer1::IShuffleLayer* shuff2 =  network->addShuffle(*conv1->getOutput(0));
    shuff2->setReshapeDimensions(nvinfer1::Dims3{0,4,grid});

    return shuff2;
}

                    

