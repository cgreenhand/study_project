//  yolov8的build实现   不考虑seg obb pose等
#pragma once

namespace yoloConfig{
    inline const char* kInputTensorName = "images";
    inline const char* kOutputTensorName = "output";
    inline const int kClsNum = 80;
    
}
