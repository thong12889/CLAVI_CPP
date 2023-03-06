#include <iostream>
#include <vector>
#include "InstanceSegmentation.h"


InstanceSegmentation::InstanceSegmentation(){}
template <typename T> T vectorProduct(std::vector<T>& v){
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}
InstanceSegmentation::InstanceSegmentation(std::string modelFilepath, std::string labelFilepath){
        this->modelFilepath = modelFilepath;
        this->classes_label_ = this->FindIndexClass(labelFilepath);
        this->num_classes_ = classes_label_.size();
}

std::vector<std::string> InstanceSegmentation::FindIndexClass(std::string &path){
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

void InstanceSegmentation::UseCUDA(){
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = 0;
		cuda_options.arena_extend_strategy = 0;
		cuda_options.cuda_mem_limit = static_cast<int>(SIZE_MAX * 1024 * 1024);
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch();
		cuda_options.do_copy_in_default_stream = 1;
		this->session_options.AppendExecutionProvider_CUDA(cuda_options);
		this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

void InstanceSegmentation::UseCPU(){
        std::cout << "Inference Execution Provider: CPU" << std::endl;
        this->session_options.SetIntraOpNumThreads(1);
        this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

Ort::Session InstanceSegmentation::SessionInit()
                {
                        Ort::Session session(this->env, modelFilepath.c_str(), this->session_options);
                        Ort::AllocatorWithDefaultOptions allocator;
                        this->numInputNodes = session.GetInputCount();
                        this->numOutputNodes = session.GetOutputCount();
                        const char* inputName = session.GetInputName(0, allocator);
                        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
                        Ort::Unowned<Ort::TensorTypeAndShapeInfo> inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
                        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
                        std::vector<int64_t> getInputDims = inputTensorInfo.GetShape();
                        if(getInputDims[2] == -1 && getInputDims[3] == -1)
                        {
                                this->inputDims.push_back(1);
                                this->inputDims.push_back(3);
                                this->inputDims.push_back(224);
                                this->inputDims.push_back(224);
                        }
                        else
                        {
                        inputDims = getInputDims;
                        }
                        const char* outputName0 = session.GetOutputName(0, allocator);
                        const char* outputName1 = session.GetOutputName(1 , allocator);
                        const char* outputName2 = session.GetOutputName(2 , allocator);
                        this->inputNames.push_back(inputName);
                        this->outputNames.push_back(outputName0);
                        this->outputNames.push_back(outputName1);
                        this->outputNames.push_back(outputName2);
                        this->inputTensorSize = vectorProduct(this->inputDims);
                        this->inputTensorValues.push_back(this->inputTensorSize);
                        this->session_ptr_ = &session;
                        return session;
                }
void InstanceSegmentation::Inference(cv::Mat &preprocessedImage)
{
        this->inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
        this->inputTensors.push_back(Ort::Value::CreateTensor<float>(this->memoryInfo, this->inputTensorValues.data(), this->inputTensorSize, this->inputDims.data(), this->inputDims.size()));
        std::cout << this->outputNames[0] << std::endl;
        std::vector<Ort::Value> outputTensorValues = (*this->session_ptr_).Run(Ort::RunOptions{nullptr}, this->inputNames.data(), this->inputTensors.data(), this->numInputNodes, this->outputNames.data(), this->numOutputNodes);
        std::cout << this->outputNames.size() << std::endl;
        const float* pred = outputTensorValues[0].GetTensorMutableData<float>();
        std::vector<int64_t> pred_dim = outputTensorValues[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<float> pred_list;

        for(int i = 0; i < pred_dim[1]; i++)
        {
                pred_list.push_back(pred[i]);
        }
        std::vector<float>::iterator iter = std::max_element(pred_list.begin(), pred_list.end());
        this->index_result_ = std::distance(pred_list.begin(), iter);
        std::cout << "Result: " << classes_label_[index_result_] << std::endl;
        std::cout << "Score : " << pred[index_result_]<< std::endl;
}

void InstanceSegmentation::DrawResult(cv::Mat &image){
        cv::putText(image, classes_label_[index_result_], cv::Point(0,20) , cv::FONT_HERSHEY_SIMPLEX, 1,cv::Scalar(0,255,0) , false );
}
