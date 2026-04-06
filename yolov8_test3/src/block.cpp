#include "block.h"
#include <math.h>
#include <cassert>




// 实现batchnorm 和 silu
static nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                        std::map<std::string, nvinfer1::Weights>& weightMap, std::string lname, float eps)
{
    const float* running_mean = static_cast<const float*>(weightMap[lname+".running_mean"].values);
    const float* running_var = static_cast<const float*>(weightMap[lname+".running_var"].values);;
    const float* gamma = static_cast<const float*>(weightMap[lname+".running_weight"].values);;
    const float* beta = static_cast<const float*>(weightMap[lname+".running_bias"].values);;
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

    nvinfer1::IScaleLayer* output = network->addScale(input,nvinfer1::ScaleMode::kCHANNEL,shift,scale,power);

    weightMap[lname+ ".shift"] =shift;
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".power"] = power;
  
    assert(output && "addBatchNorm2d error");
    return output;



}

static inline nvinfer1::IElementWiseLayer* addSiLU(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input)
{
    nvinfer1::IActivationLayer* sigmoid = network->addActivation(input,nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* output = network->addElementWise(*sigmoid->getOutput(0),input,nvinfer1::ElementWiseOperation::kPOW);

    assert(output && "siLU error");
    return output;
}

static nvinfer1::IElementWiseLayer* addBottleNeck(nvinfer1::INetworkDefinition* network,nvinfer1::ITensor& input,
                                                    std::map<std::string, nvinfer1::Weights>& weightMap, int ch1, int ch2 , bool shortcut, float e, std::string lname)
{

    int c_ = int (ch2 * e);

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLu(network,input,weightMap,c_,3,1,1,lname+".cv1");
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

nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weigthMap, int c_in, int c_out, int nd,
                                bool shortcut, float e, std::string lname)
{   
    int c_ = int((float)c_out * e);
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLu(network,input,weigthMap,c_out,1,1,0,lname+"cv1");
    assert(conv1 && "C2F -> conv1 error");

    nvinfer1::Dims d = conv1->getOutput(0)->getDimensions();

    nvinfer1::ISliceLayer* split1 = network->addSlice(*conv1->getOutput(0),nvinfer1::Dims3{0,0,0},nvinfer1::Dims{d.d[0]/2,d.d[1],d.d[2]},nvinfer1::Dims{1,1,1});
    nvinfer1::ISliceLayer* split2 = network->addSlice(*conv1->getOutput(0),nvinfer1::Dims3{d.d[0]/2,0,0},nvinfer1::Dims{d.d[0]/2,d.d[1],d.d[2]},nvinfer1::Dims{1,1,1});

    nvinfer1::ITensor * bottlencek_in = split2->getOutput(0);
    nvinfer1::ITensor* inputTensor0[] = {split1->getOutput(0),split2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat1 = network->addConcatenation(inputTensor0,2);

    for(int i = 0 ; i < nd ; i++){
        nvinfer1::IElementWiseLayer* bottleneck = addBottleNeck(network,*bottlencek_in,weigthMap,c_,c_,true,0.5,lname+".m."+std::to_string(i));
        bottlencek_in = bottleneck->getOutput(0);

        nvinfer1::ITensor* inputTensors[] = {cat1->getOutput(0),bottlencek_in};
        cat1 = network->addConcatenation(inputTensors,2);
    }

    nvinfer1::IElementWiseLayer* conv2 = convBnSiLu(network,*cat1->getOutput(0),weigthMap,c_out,1,1,0,lname+"cv2");

    assert(conv2 && "C2F -> conv2 error");

    return conv2; 
}

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weightMap,int c_in, int c_out, int k, std::string lname)
{
    int c_ = c_out / 2;
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLu(network,input,weightMap,c_,1,1,0,lname+".cv1");
    nvinfer1::IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0),nvinfer1::PoolingType::kMAX,nvinfer1::Dims{k,k});
    pool1->setStrideNd(nvinfer1::DimsHW{1,1});
    pool1->setPaddingNd(nvinfer1::DimsHW{k/2,k/2});

    nvinfer1::IPoolingLayer* pool2 = network->addPoolingNd(*pool1->getOutput(0),nvinfer1::PoolingType::kMAX,nvinfer1::Dims{k,k});
    pool2->setStrideNd(nvinfer1::DimsHW{1,1});
    pool2->setPaddingNd(nvinfer1::DimsHW{k/2,k/2});

    nvinfer1::IPoolingLayer* pool3 = network->addPoolingNd(*pool2->getOutput(0),nvinfer1::PoolingType::kMAX,nvinfer1::Dims{k,k});
    pool3->setStrideNd(nvinfer1::DimsHW{1,1});
    pool3->setPaddingNd(nvinfer1::DimsHW{k/2,k/2});

    nvinfer1::ITensor* inputTensors[] = {conv1->getOutput(0),pool1->getOutput(0),pool2->getOutput(0),pool3->getOutput(0)};

    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors,4);

    nvinfer1::IElementWiseLayer* conv2 = convBnSiLu(network,*cat->getOutput(0),weightMap,c_out,1,1,0,lname+"cv2");

    assert(conv2 && "SPPF -> conv2 error");

    return conv2;


}

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weightMap,
                                int ch, int grid, int k , int s, int p, std::string lname)
{

    //  64,grid
    nvinfer1::IShuffleLayer* shuff1 = network->addShuffle(input);
    shuff1->setReshapeDimensions(nvinfer1::Dims3{4,16,grid});
    shuff1->setSecondTranspose(nvinfer1::Permutation{1,0,2});
    //  16,4,frid
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*shuff1->getOutput(0));

    nvinfer1::Weights bias = {nvinfer1::DataType::kFLOAT,nullptr,0};
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolution(*softmax->getOutput(0),1,nvinfer1::DimsHW{1,1},weightMap[lname],bias);
    conv1->setStrideNd(nvinfer1::DimsHW{s,s});
    conv1->setPaddingNd(nvinfer1::DimsHW{p,p});

    nvinfer1::IShuffleLayer* shuff2 = network->addShuffle(*conv1->getOutput(0));
    shuff2->setReshapeDimensions(nvinfer1::Dims2{4,grid});

    assert(shuff2 && "DFL shuff2 error");
    
    return shuff2;

}

// DFL ->  4,8400  4+cls,8400  4代表在原特征图上的偏移？
nvinfer1::IPluginV2Layer* addYoloLayer()
{

}