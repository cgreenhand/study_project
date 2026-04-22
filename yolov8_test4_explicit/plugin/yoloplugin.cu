#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "cuda_utils.h"
#include "types.h"
#include "yoloplugin.h"


namespace {
    static const char* YOLO_PLUGIN_NAME{"YoloPlugin_Dynamic"};
    static const char* YOLO_PLUGIN_VERSION{"1"};
}

namespace Tn {

    template <typename T>
    void write(char*& buffer, const T& val) {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(const char*& buffer, T& val) {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}  // namespace Tn

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}


namespace nvinfer1{


// lifecycle

// 正常初始化
YoloPlugin::YoloPlugin(int classCount, int numberofpoints, float confthreshkeypoints, int netWidth, int netHeight,
                       int maxOut, bool is_segmentation, bool is_pose, bool is_obb, 
                       const int* strides, int stridesLength)
    : mClassCount(classCount)
    , mNumberofpoints(numberofpoints)
    , mConfthreshkeypoints(confthreshkeypoints)
    , mYoloV8netHeight(netHeight)
    , mYoloV8NetWidth(netWidth)
    , mMaxOutObject(maxOut)
    , is_segmentation_(is_segmentation)
    , is_pose_(is_pose)
    , is_obb_(is_obb)
    , mStridesLength(stridesLength)
    , mDeviceStrides(nullptr) 
{
    // 只有必须申请内存的操作留在函数体内，并加上空指针保护
    mStrides = new int[mStridesLength];
    if (strides != nullptr) {
        std::copy(strides, strides + stridesLength, mStrides); // 使用 std::copy 替代 memcpy 更具 C++ 风格
    }
}

// 从文件数据中反序列化      
YoloPlugin::YoloPlugin(const void* data, size_t length)
{
    using namespace Tn;
    const char * d = reinterpret_cast<const char*>(data), *a = d;
    read(d,mClassCount);
    read(d,mNumberofpoints);
    read(d,mConfthreshkeypoints);
    read(d,mYoloV8NetWidth);
    read(d,mYoloV8netHeight);
    read(d,mMaxOutObject);
    read(d,is_segmentation_);
    read(d,is_pose_);
    read(d,is_obb_);
    read(d,mStridesLength);
    mStrides = new int[mStridesLength];
    for(int i = 0; i < mStridesLength; i++){
        read(d,mStrides[i]);
    }
    mDeviceStrides =nullptr;



}

YoloPlugin::~YoloPlugin(){

    if(mStrides != nullptr){
        delete[] mStrides;
        mStrides = nullptr;
    }
}

YoloPlugin::YoloPlugin(const YoloPlugin& other): mClassCount(other.mClassCount)
    , mNumberofpoints(other.mNumberofpoints)
    , mConfthreshkeypoints(other.mConfthreshkeypoints)
    , mYoloV8NetWidth(other.mYoloV8NetWidth)
    , mYoloV8netHeight(other.mYoloV8netHeight)
    , mMaxOutObject(other.mMaxOutObject)
    , is_segmentation_(other.is_segmentation_)
    , is_pose_(other.is_pose_)
    , is_obb_(other.is_obb_)
    , mStridesLength(other.mStridesLength)
    , mPluginNamespace(other.mPluginNamespace){

    if (other.mStrides != nullptr) {
        mStrides = new int[mStridesLength];
        memcpy(mStrides, other.mStrides, mStridesLength * sizeof(int));
    }
    mDeviceStrides = nullptr;
    mThreadCount = other.mThreadCount;

}


IPluginV2DynamicExt* YoloPlugin::clone() const noexcept{
    try{
        auto* plugin = new YoloPlugin(*this);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch(const std::exception& e){
        return nullptr;
    }

};

// basic info
AsciiChar const* YoloPlugin::getPluginType() const noexcept
{
    return YOLO_PLUGIN_NAME;
}

AsciiChar const* YoloPlugin::getPluginVersion() const noexcept
{
    return YOLO_PLUGIN_VERSION;
}

int32_t YoloPlugin::getNbOutputs() const noexcept
{
    return 1;
}

void YoloPlugin::setPluginNamespace (const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

AsciiChar const* YoloPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// brain
DimsExprs YoloPlugin::getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // 确保单个输出  n c h w
    assert(outputIndex == 0);
    assert(inputs[0].nbDims == 4);
    nvinfer1::DimsExprs output;

    output.nbDims = 2;
    output.d[0] = inputs[0].d[0];
    int col_len = yoloConfig::kMaxNumOutputBbox * (sizeof(Detection) / sizeof(float)) + 1;
    output.d[1] = exprBuilder.constant(col_len);
    return output;
}

// TODO 暂时只支持FP32  后续可以调整为支持FP16
bool YoloPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    assert(pos < nbInputs+nbOutputs);

    bool isFloat32 = (inOut[pos].type == DataType::kFLOAT);
    bool isLinear = (inOut[pos].format == TensorFormat::kLINEAR);

    return isFloat32 && isLinear;

}

nvinfer1::DataType  YoloPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return nvinfer1::DataType::kFLOAT;
}




void YoloPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    assert(nbInputs >=1);
    auto const& inputDims = in[0].desc.dims;
    mYoloV8netHeight = inputDims.d[2];
    mYoloV8NetWidth = inputDims.d[3];
}


// muscle
int32_t YoloPlugin::initialize() noexcept
{
    if(mDeviceStrides == nullptr && mStrides!=nullptr){
        cudaMalloc((void**)&mDeviceStrides, mStridesLength*sizeof(int));
        cudaMemcpy(mDeviceStrides,mStrides,mStridesLength,cudaMemcpyHostToDevice);
    }
    return 0;
}

void YoloPlugin::terminate() noexcept
{
    if(mDeviceStrides != nullptr){
        cudaFree(mDeviceStrides);
        mDeviceStrides = nullptr;
    }
}

size_t YoloPlugin::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept
{
    return 0;
}        


int32_t YoloPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

}


// persistence
// private 一共占多少字节
size_t YoloPlugin::getSerializationSize() const noexcept
{
    return sizeof(mClassCount) + 
           sizeof(mNumberofpoints) + 
           sizeof(mConfthreshkeypoints) + 
           sizeof(mYoloV8NetWidth) + 
           sizeof(mYoloV8netHeight) + 
           sizeof(mMaxOutObject) + 
           sizeof(is_segmentation_) + 
           sizeof(is_pose_) + 
           sizeof(is_obb_) + 
           sizeof(mStridesLength) + 
           (sizeof(int) * mStridesLength);
}

void YoloPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer);
    const char *a = d;

    Tn::write(d, mClassCount);
    Tn::write(d, mNumberofpoints);
    Tn::write(d, mConfthreshkeypoints);
    Tn::write(d, mYoloV8NetWidth);
    Tn::write(d, mYoloV8netHeight);
    Tn::write(d, mMaxOutObject);
    Tn::write(d, is_segmentation_);
    Tn::write(d, is_pose_);
    Tn::write(d, is_obb_);
    Tn::write(d, mStridesLength);

    for (int i = 0; i < mStridesLength; ++i) {
        Tn::write(d, mStrides[i]);
    }

    assert(d == a + getSerializationSize());

}

void YoloPlugin::destroy() noexcept
{
    delete this;
}


void YoloPlugin::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int mYoloV8netHeight,
                    int mYoloV8NetWidth, int batchSize)
{

}
      



PluginFieldCollection YoloPluginCreator::mFC{};
YoloPluginCreator::YoloPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("classCount",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("numberofpoints",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("confthreshkeypoints",nullptr,PluginFieldType::kFLOAT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("netHeight",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("netWidth",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("maxOut",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("is_segmentation",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("is_pose",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("is_obb",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stridesLength",nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("strides", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();

}

YoloPluginCreator::~YoloPluginCreator()
{

}

// 身份信息
const char* YoloPluginCreator::getPluginName() const noexcept
{
    return  YOLO_PLUGIN_NAME;
}

const char* YoloPluginCreator::getPluginVersion() const noexcept
{
    return YOLO_PLUGIN_VERSION;
}


const PluginFieldCollection* YoloPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

//     // 插件创建 工厂方法
IPluginV2* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
// 1. 定义默认变量（防止参数包里漏掉某个参数）
    int classCount = 80;
    int numberofpoints = 0;
    float confthreshkeypoints = 0.5f;
    int netWidth = 640;
    int netHeight = 640;
    int maxOut = 1000;
    bool is_segmentation = false;
    bool is_pose = false;
    bool is_obb = false;
    const int* strides = nullptr;
    int stridesLength = 0;

    const PluginField* fields = fc->fields;
    for(int i=0; i<fc->nbFields; i++){

        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "classCount")) {
            classCount = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "numberofpoints")) {
            numberofpoints = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "confthreshkeypoints")) {
            confthreshkeypoints = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "netWidth")) {
            netWidth = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "netHeight")) {
            netHeight = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "maxOut")) {
            maxOut = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "is_segmentation")) {
            is_segmentation = *(static_cast<const int*>(fields[i].data)) > 0;
        }
        else if (!strcmp(attrName, "is_pose")) {
            is_pose = *(static_cast<const int*>(fields[i].data)) > 0;
        }
        else if (!strcmp(attrName, "is_obb")) {
            is_obb = *(static_cast<const int*>(fields[i].data)) > 0;
        }
        else if (!strcmp(attrName, "strides")) {
            strides = static_cast<const int*>(fields[i].data);
            stridesLength = fields[i].length;
        }
    }

    YoloPlugin* plugin = new YoloPlugin(classCount, numberofpoints, confthreshkeypoints, netWidth, netHeight, 
                                 maxOut, is_segmentation, is_pose, is_obb, strides, stridesLength);

    
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;

}

nvinfer1::IPluginV2* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength )noexcept
{
    YoloPlugin* plugin = new YoloPlugin(serialData,serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

// 命名空间
void YoloPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}
const char* YoloPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

}
