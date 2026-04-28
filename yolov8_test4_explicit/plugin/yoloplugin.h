#pragma once
#include <string>
#include <vector>
#include "NvInfer.h"
#include "macros.h"

namespace nvinfer1{

class API YoloPlugin: public IPluginV2DynamicExt{

public:
    // lifecycle
    YoloPlugin(int classCount, int numberofpoints, float confthreshkeypoints, int netWidth, int netHeight,
                    int maxOut, bool is_segmentation, bool is_pose, bool is_obb, int OutputElem, const int* strides, int stridesLength);

    YoloPlugin(const void* data, size_t length);
    // 拷贝构造函数
    YoloPlugin(const YoloPlugin& other);

    ~YoloPlugin();
    IPluginV2DynamicExt* clone() const noexcept override;

    // basic info
    AsciiChar const* getPluginType() const noexcept override;
    AsciiChar const* getPluginVersion() const noexcept override ;
    int32_t getNbOutputs() const noexcept override;
    void setPluginNamespace (const char* pluginNamespace) noexcept override;
    AsciiChar const* getPluginNamespace() const noexcept override;

    // brain
    virtual DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

    virtual bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    virtual nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    virtual void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;


    // muscle
    virtual int32_t initialize() noexcept override;
    virtual void terminate() noexcept override;
    virtual size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;

    virtual int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;


    // persistence
    // private 一共占多少字节
    virtual size_t getSerializationSize() const noexcept override;
    virtual void serialize(void* buffer) const noexcept override;
    virtual void destroy() noexcept override;

private:
    void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int mYoloV8netHeight,
                    int mYoloV8NetWidth, int batchSize);
    int mThreadCount = 256;
    std::string mPluginNamespace; 
    int mClassCount;
    int mNumberofpoints;
    float mConfthreshkeypoints;
    int mYoloV8NetWidth;
    int mYoloV8netHeight;
    int mMaxOutObject;
    bool is_segmentation_;
    bool is_pose_;
    bool is_obb_;
    int mOutputElem;
    int mStridesLength;
    int* mStrides;
    int* mDeviceStrides;
    
};


class API YoloPluginCreator: public IPluginCreator{

public:
    YoloPluginCreator();
    ~YoloPluginCreator() override = default;

    // 身份信息
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    // 插件创建 工厂方法
    nvinfer1::IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength )noexcept override;

    // 命名空间
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;

};

REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

}