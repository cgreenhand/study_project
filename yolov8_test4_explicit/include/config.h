//  yolov8的build实现   不考虑seg obb pose等
#pragma once

namespace yoloConfig{
    inline const char* kInputTensorName = "images";
    inline const char* kOutputTensorName = "output";
    inline const int kClsNum = 80;
    inline const int kMinBatch = 1;
    inline const int kMaxBatch = 1;
    inline const int kOptBatch = 1;
    inline const int kInputSize = 640; 
    inline const int kNumberOfPoints = 17;
    inline const int kConfThreshKeypoints = 0.5;
    inline const int kMaxNumOutputBbox = 16800;
    //  

    
}


    // combinedInfo[0] = num_class;
    // combinedInfo[1] = kNumberOfPoints;
    // combinedInfo[2] = kConfThreshKeypoints;
    // combinedInfo[3] = kInputW;
    // combinedInfo[4] = kInputH;
    // combinedInfo[5] = kMaxNumOutputBbox;
    // combinedInfo[6] = is_segmentation;
    // combinedInfo[7] = is_pose;
    // combinedInfo[8] = is_obb;