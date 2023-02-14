#include<iostream>
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


class ObjectClassification{
    public:
        ObjectClassification();
        ObjectClassification(std::string , std::string);
        std::vector<std::string> FindIndexClass(std::string&);
		void UseCUDA(Ort::SessionOptions &);
		void UseCPU(Ort::SessionOptions &);
		Ort::Session SessionInit(Ort::Env &env, 
                        Ort::SessionOptions &session_options, 
                        size_t &numInputNodes ,
                        size_t &numOutputNodes,
                        std::vector<int64_t> &inputDims,
                        std::vector<const char*> &inputNames,
                        std::vector<const char*> &outputNames,
                        std::vector<Ort::Value> &inputTensors,
                        size_t &inputTensorSize,
                        std::vector<float> &inputTensorValues
                        );

        
    private:
        std::string modelFilepath;
        std::vector<std::string> classes_label_;
        int num_classes_;
        Ort::Session *session_ptr_;
        
};