#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "cuda_utils.h"
#include "types.h"
#include "yololayer.h"
#include "config.h"
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
    if(idx >= numElements) return;

    const int N_kpts = nk;
    int total_grid = grid_h*grid_w;
    int info_len = 4 + classes + (is_segmentation ? 32 : 0) + (is_pose ? N_kpts*3:0) + (is_obb ? 1:0);
    int batchIdx = idx / total_grid;
    int elemIdx = idx % total_grid;
    const float* curInput = input + batchIdx * total_grid * info_len;
    int outputIdx = batchIdx * outputElem;

    int class_id = 0;
    float max_cls_prob = 0.0;
    for(int i=4; i<4+classes; i++){
        float p = Logist(curInput[elemIdx+i*total_grid]);

        if(p>max_cls_prob){
            max_cls_prob = p;
            class_id = i -4;
        }
    }

    if(max_cls_prob < 0.1) return ;

    int count = (int)atomicAdd(output+outputIdx,1);
    if(count >= maxoutobject) return ;

    char* data = (char*)(output+outputIdx) + sizeof(float) + count * sizeof(Detection);

    Detection* det = (Detection*)data;

    int row = elemIdx / grid_w;
    int col = elemIdx % grid_w;

    det->conf = max_cls_prob;
    det->class_id = class_id;
    
    det->bbox[0] = (col+0.5f-curInput[elemIdx + 0*total_grid]) * stride;
    det->bbox[1] = (row + 0.5f - curInput[elemIdx + 1*total_grid]) * stride;
    det->bbox[2] = (col+0.5f + curInput[elemIdx+2*total_grid]) * stride;
    det->bbox[3] = (row + 0.5f + curInput[elemIdx + 3*total_grid])*stride;

    if(is_segmentation){
        for(int k = 0; k < 32; k++){
            curInput[elemIdx + (4 + classes + (is_pose ? N_kpts * 3 : 0) + (is_obb ? 1 : 0) + k) * total_grid];
        }
    }

    if (is_pose) {
        for (int kpt = 0; kpt < N_kpts; kpt++) {
            int kpt_x_idx = (4 + classes + (is_segmentation ? 32 : 0) + (is_obb ? 1 : 0) + kpt * 3) * total_grid;
            int kpt_y_idx = (4 + classes + (is_segmentation ? 32 : 0) + (is_obb ? 1 : 0) + kpt * 3 + 1) * total_grid;
            int kpt_conf_idx = (4 + classes + (is_segmentation ? 32 : 0) + (is_obb ? 1 : 0) + kpt * 3 + 2) * total_grid;

            float kpt_confidence = Logist(curInput[elemIdx + kpt_conf_idx]);

            float kpt_x = (curInput[elemIdx + kpt_x_idx] * 2.0 + col) * stride;
            float kpt_y = (curInput[elemIdx + kpt_y_idx] * 2.0 + row) * stride;

            bool is_within_bbox =
                    kpt_x >= det->bbox[0] && kpt_x <= det->bbox[2] && kpt_y >= det->bbox[1] && kpt_y <= det->bbox[3];

            if (kpt_confidence < confkeypoints || !is_within_bbox) {
                det->keypoints[kpt * 3] = -1;
                det->keypoints[kpt * 3 + 1] = -1;
                det->keypoints[kpt * 3 + 2] = -1;
            } else {
                det->keypoints[kpt * 3] = kpt_x;
                det->keypoints[kpt * 3 + 1] = kpt_y;
                det->keypoints[kpt * 3 + 2] = kpt_confidence;
            }
        }
    }

    if (is_obb) {
        double pi = M_PI;
        auto angle_inx = curInput[elemIdx + (4 + classes + (is_segmentation ? 32 : 0) + (is_pose ? N_kpts * 3 : 0) +
                                             0) * total_grid];
        auto angle = (Logist(angle_inx) - 0.25f) * pi;

        auto cos1 = cos(angle);
        auto sin1 = sin(angle);
        auto xf = (curInput[elemIdx + 2 * total_grid] - curInput[elemIdx + 0 * total_grid]) / 2;
        auto yf = (curInput[elemIdx + 3 * total_grid] - curInput[elemIdx + 1 * total_grid]) / 2;

        auto x = xf * cos1 - yf * sin1;
        auto y = xf * sin1 + yf * cos1;

        float cx = (col + 0.5f + x) * stride;
        float cy = (row + 0.5f + y) * stride;

        float w1 = (curInput[elemIdx + 0 * total_grid] + curInput[elemIdx + 2 * total_grid]) * stride;
        float h1 = (curInput[elemIdx + 1 * total_grid] + curInput[elemIdx + 3 * total_grid]) * stride;
        det->bbox[0] = cx;
        det->bbox[1] = cy;
        det->bbox[2] = w1;
        det->bbox[3] = h1;
        det->angle = angle;
    }
    

}


// ++++++++++++++++++++++
// YoloLayerPlugin 实现
// ++++++++++++++++++++++
//  自己填入权重
YoloLayerPlugin::YoloLayerPlugin(int classCount, int numberofpoints, float confthreshkeypoints, int netWidth, int netHeight,
                int maxOut, bool is_segmentation, bool is_pose, bool is_obb, const int* strides, int stridesLength){

    mClassCount = classCount;
    mNumberofpoints = numberofpoints;
    mConfthreshkeypoints = confthreshkeypoints;
    mYoloV8NetWidth = netWidth;
    mYoloV8NetHeight = netHeight;
    mMaxOutObject = maxOut;
    mStridesLength = stridesLength;
    mStrides = new int[stridesLength];
    memcpy(mStrides, strides, stridesLength * sizeof(int));
    is_segmentation_ = is_segmentation;
    is_pose_ = is_pose;
    is_obb_ = is_obb;

}


// 从文件中反序列
YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length){

    using namespace Tn;
    const char* d = reinterpret_cast<const char*>(data), *a = d;
    read(d,mClassCount);
    read(d,mNumberofpoints);
    read(d,mConfthreshkeypoints);
    read(d,mThreadCount);
    read(d,mYoloV8NetWidth);
    read(d,mYoloV8NetHeight);
    read(d,mMaxOutObject);
    read(d,mStridesLength);
    mStrides = new int[mStridesLength];
    for(int i = 0; i <mStridesLength; i++){
        read(d,mStrides[i]);
    }
    read(d,is_segmentation_);
    read(d,is_pose_);
    read(d,is_obb_);

    assert(d==a+length);

}

YoloLayerPlugin::~YoloLayerPlugin(){
    if(mStrides != nullptr){
        delete[] mStrides;
        mStrides = nullptr;
    }
}


nvinfer1::Dims YoloLayerPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) TRT_NOEXCEPT{
    int total_size = mMaxOutObject * sizeof(Detection) / sizeof(float);
    return nvinfer1::Dims3(total_size+1,1,1);
}

int YoloLayerPlugin::initalize() TRT_NOEXCEPT{
    return 0;
}

int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace,
                cudaStream_t stream) TRT_NOEXCEPT{
    

}

size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
    return sizeof(mClassCount) + sizeof(mNumberofpoints) + sizeof(mConfthreshkeypoints) + sizeof(mThreadCount) +
           sizeof(mYoloV8NetHeight) + sizeof(mYoloV8NetWidth) + sizeof(mMaxOutObject) + sizeof(mStridesLength) +
           sizeof(int) * mStridesLength + sizeof(is_segmentation_) + sizeof(is_pose_) + sizeof(is_obb_);

}

void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT{
    using namespace Tn;
    char *d = static_cast<char*>(buffer), *a = d;
    Tn::write(d, mClassCount);
    Tn::write(d, mNumberofpoints);
    Tn::write(d, mConfthreshkeypoints);
    Tn::write(d, mThreadCount);
    Tn::write(d, mYoloV8NetWidth);
    Tn::write(d, mYoloV8NetHeight);
    Tn::write(d, mMaxOutObject);
    Tn::write(d, mStridesLength);

    for(int i = 0; i<mStridesLength; ++i){
        Tn::write(d,mStrides[i]);
    }

    Tn::write(d,is_segmentation_);
    Tn::write(d,is_pose_);
    Tn::write(d,is_obb_);

}


const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT {
    return "YoloLayer_TRT";
}



const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT {
    return "1";
}


void YoloLayerPlugin::destory() TRT_NOEXCEPT {
    delete this;
}



IPluginV2IOExt*  YoloLayerPlugin::clone() const TRT_NOEXCEPT {
    YoloLayerPlugin* p =
            new YoloLayerPlugin(mClassCount, mNumberofpoints, mConfthreshkeypoints, mYoloV8NetWidth, mYoloV8netHeight,
                                mMaxOutObject, is_segmentation_, is_pose_, is_obb_, mStrides, mStridesLength);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}



void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT {
    return mPluginNamespace;
}


nvinfer1::DataType YoloLayerPlugin::getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const TRT_NOEXCEPT {

    return nvinfer1::DataType::kFLOAT;            
}



bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
                                      int nbInputs) const TRT_NOEXCEPT {
    return false;                                 
}


bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT {
    return false;
}


void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
                         IGpuAllocator* gpuAllocator) TRT_NOEXCEPT{
                        
}

void configurePlugin(PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out,
                         int32_t nbOutput) TRT_NOEXCEPT {
                        
}


void detachFromContext() TRT_NOEXCEPT {

}

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