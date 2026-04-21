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
                int maxOut, bool is_segmentation, bool is_pose, bool is_obb, const int* strides, int stridesLength)
{
    mClassCount = classCount;
    mNumberofpoints = numberofpoints;
    mConfthreshkeypoints = confthreshkeypoints;
    mYoloV8netHeight = netHeight;
    mYoloV8NetWidth = netWidth;

    mMaxOutObject = maxOut;
    is_segmentation_ = is_segmentation;
    is_pose_ = is_pose;
    is_obb_ = is_obb;
    mStridesLength = stridesLength;
    mStrides = new int[stridesLength];
    memcpy(mStrides, strides, stridesLength * sizeof(int));


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

//     // brain
//     virtual DimsExprs getOutputDimensions(
//         int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

//     virtual bool supportsFormatCombination(
//         int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

//     virtual nvinfer1::DataType getOutputDataType(
//         int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
//     virtual void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
//         DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;


//     // muscle
//     virtual int32_t initialize() noexcept override;
//     virtual void terminate() noexcept override;
//     virtual size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
//         int32_t nbOutputs) const noexcept override;

//     virtual int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
//         void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;


//     // persistence
//     // private 一共占多少字节
//     virtual size_t getSerializationSize() const noexcept override;
//     virtual void serialize(void* buffer) const noexcept override;
//     virtual void destroy() noexcept override;


//     void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int mYoloV8netHeight,
//                     int mYoloV8NetWidth, int batchSize);
      
// };



//     YoloPluginCreator();
//     ~YoloPluginCreator() override = default;

//     // 身份信息
//     const char* getPluginName() const noexcept override;
//     const char* getPluginVersion() const noexcept override;
//     const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

//     // 插件创建 工厂方法
//     nvinfer1::IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
//     nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength )noexcept override;

//     // 命名空间
//     void setPluginNamespace(const char* libNamespace) noexcept override;
//     const char* getPluginNamespace() const noexcept override;


//     static PluginFieldCollection mFC;
//     static std::vector<PluginField> mPluginAttributes;
//     std::string mNamespace;



}
