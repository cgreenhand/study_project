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


// ++++++++++++++++++++++
// cuda_kernel 实现
// ++++++++++++++++++++++

__device__ float Logist(float data){
    return 1.0/(1.0f+expf(-data));
};

__global__ void CalDetection(const float* input, float* output, int numElements, int maxoutobject, const int grid_h,
                            int grid_w, const int stride, int classes, int nk, float confkeypoints, int outputElem,
                            bool is_segmentation, bool is_pose, bool is_obb){
    
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    

}


// ++++++++++++++++++++++
// YoloLayerPlugin 实现
// ++++++++++++++++++++++







// ++++++++++++++++++++++
// YoloLayerCreator 实现
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

IPluginV2IOExt* YoloLayerCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT{

    assert(fc->nbFields == 1);
    assert(strcmp(fc->fields[0].name, "combinedInfo") == 0);
    const int* combinedInfo = static_cast<const int*>(fc->fields[0].data);
    int netinfo_count = 9;
    int class_count = combinedInfo[0];
    int numberofpoints = combinedInfo[1];
    float confthreshkeypoints = combinedInfo[2];
    int input_w = combinedInfo[3];
    int input_h = combinedInfo[4];
    int max_output_object_count = combinedInfo[5];
    bool is_segmentation = combinedInfo[6];
    bool is_pose = combinedInfo[7];
    bool is_obb = combinedInfo[8];
    const int* px_arry = combinedInfo + netinfo_count;
    int px_arry_length = fc->fields[0].length - netinfo_count;
    YoloLayerPlugin* obj = 
            new YoloLayerPlugin(class_count,numberofpoints,confthreshkeypoints,input_w,input_h,
                            max_output_object_count, is_segmentation, is_pose,is_obb, px_arry, px_arry_length);
    obj->setPluginNamespace(mNamespace.c_str());

    return nullptr;

}






}  // namespace nvinfer1