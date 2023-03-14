#include<iostream>
#include <opencv2/core/matx.hpp>
#include <vector>

//ONNX RUNTIME LIB
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>

//OPENCV LIB
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iterator>
#include <memory>

#include "Instance.h"


class InstanceSegmentation{
    public:
        InstanceSegmentation();
        InstanceSegmentation(std::string , std::string);
	void UseCUDA();
	void UseCPU();
	Ort::Session SessionInit();
        void Inference(cv::Mat &preprocessedImage);
        void DrawResult(cv::Mat &image);

        
    private:
        std::vector<std::string> FindIndexClass(std::string&);
        void get_candidates();
        std::string modelFilepath;
        std::vector<std::string> classes_label_;
        int num_classes_;
        size_t index_result_;

        Ort::Session *session_ptr_;
        Ort::Env env;
        Ort::SessionOptions session_options;
        size_t numInputNodes;
        size_t numOutputNodes;
        std::vector<int64_t> inputDims;
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;
        std::vector<Ort::Value> inputTensors;
        size_t inputTensorSize;
        std::vector<float> inputTensorValues;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        const float* pred_; 
        std::vector<int64_t> pred_dim; 
        const int64_t* label_value_; 
        const float* seg_;
        std::vector<int64_t> seg_dim_;
        std::vector<Instance> instances_;
        std::vector<cv::Vec3b> color_list;
        float bbox_thresh_ = 0.5;
        float mask_threh_ = 0.5;
        float opacity_ = 0.6;
};
