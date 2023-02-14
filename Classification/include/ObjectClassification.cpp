#include <iostream>
#include <vector>
#include "ObjectClassification.h"


ObjectClassification::ObjectClassification(){}
template <typename T> T vectorProduct(std::vector<T>& v){
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}
ObjectClassification::ObjectClassification(std::string modelFilepath, std::string labelFilepath){
        this->modelFilepath = modelFilepath;
        this->classes_label_ = this->FindIndexClass(labelFilepath);
        this->num_classes_ = classes_label_.size();
}

std::vector<std::string> ObjectClassification::FindIndexClass(std::string &path){
        std::vector<std::string> path_temp;
        std::ifstream input(path);
        for( std::string line; getline( input, line ); )
        {
                //Fillter Unknown Character
                for(int i =0 ; i<line.length(); i++){
                        if((int)line[i] < 32){
                                line.erase(i,1);
                        }
                }
                path_temp.push_back(line);
        }
        return path_temp;
}

void ObjectClassification::UseCUDA(Ort::SessionOptions& session_options){
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = 0;
		cuda_options.arena_extend_strategy = 0;
		cuda_options.cuda_mem_limit = static_cast<int>(SIZE_MAX * 1024 * 1024);
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch();
		cuda_options.do_copy_in_default_stream = 1;
		session_options.AppendExecutionProvider_CUDA(cuda_options);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

void ObjectClassification::UseCPU(Ort::SessionOptions& session_options){
        std::cout << "Inference Execution Provider: CPU" << std::endl;
		session_options.SetIntraOpNumThreads(1);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

Ort::Session ObjectClassification::SessionInit(Ort::Env &env, 
                        Ort::SessionOptions &session_options, 
                        size_t &numInputNodes ,
                        size_t &numOutputNodes,
                        std::vector<int64_t> &inputDims,
                        std::vector<const char*> &inputNames,
                        std::vector<const char*> &outputNames,
                        std::vector<Ort::Value> &inputTensors,
                        size_t &inputTensorSize,
                        std::vector<float> &inputTensorValues
                        ){
                            Ort::Session session(env, modelFilepath.c_str(), session_options);
                            Ort::AllocatorWithDefaultOptions allocator;
                            numInputNodes = session.GetInputCount();
                            numOutputNodes = session.GetOutputCount();
                            const char* inputName = session.GetInputName(0, allocator);
                            Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
                            Ort::Unowned<Ort::TensorTypeAndShapeInfo> inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
                            ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
                            std::vector<int64_t> getInputDims = inputTensorInfo.GetShape();
                            if(getInputDims[2] == -1 && getInputDims[3] == -1)
                            {
                                inputDims.push_back(1);
                                inputDims.push_back(3);
                                inputDims.push_back(224);
                                inputDims.push_back(224);
                            }
                            else
                            {
                                inputDims = getInputDims;
                            }
                            const char* outputName0 = session.GetOutputName(0, allocator);
                            inputNames.push_back(inputName);
                            outputNames.push_back(outputName0);
                            inputTensorSize = vectorProduct(inputDims);
                            inputTensorValues.push_back(inputTensorSize);

                            return session;
                        }

