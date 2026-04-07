#include "NvInfer.h"
#include <vector>
#include <map>



// conv bn SiLu

nvinfer1::IElementWiseLayer* convBnSiLu(nvinfer1::INetworkDefinition* network, 
                                        nvinfer1::ITensor& input, std::map<std::string, nvinfer1::Weights>& weightMap,
                                        int ch, int k, int s, int p, std::string lname);

nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weigthMap, int c_in, int c_out, int nd,
                                bool shortcut, float e, std::string lname);

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weightMap,int c_in, int c_out, int k, std::string lname);

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                std::map<std::string, nvinfer1::Weights>& weightMap,
                                int ch, int grid, int k , int s, int p, std::string lname) ;
                            
nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
                                       std::vector<nvinfer1::IConcatenationLayer*> dets, const int* px_arry,
                                       int px_arry_num, int num_class, bool is_segmentation, bool is_pose,
                                       bool is_obb);

