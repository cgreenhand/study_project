#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "cuda_utils.h"
#include "types.h"
#include "yololayer.h"

namespace Tn{
template <typename T>
void write(char*& buffer, const T& val){
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char*&buffer, T&val){
    val = *reinterpret_cast<const T*>(buffer);
    buffer+=sizeof(T);
}


__device__ float sigmoid(float x){
    return 1.0f/(1.0f+exp(-x));
}

}


namespace nvinfer1{

// YoloLayerPlugin  cuda_kernel YoloLayerPluginCreator


//  ++++++++++++++++++++++++


// ++++++++++++++++++++++
PluginFieldCollection YoloLayerCreator::mFC{};
std::vector<PluginField> YoloLayerCreator::mPluginAttributes;
YoloLayerCreator::YoloLayerCreator(){
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloLayerCreator::getPluginName() const TRT_NOEXCEPT{
    return "YoloLayer_TRT";
}

const char* YoloLayerCreator::getPluginVersion() const TRT_NOEXCEPT{
    return "1";
}

const PluginFieldCollection* YoloLayerCreator::getFieldNames() const TRT_NOEXCEPT{
    return &mFC;
}

IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection fc) TRT_NOEXCEPT{

    

    return nullptr;

}






}  // namespace nvinfer1