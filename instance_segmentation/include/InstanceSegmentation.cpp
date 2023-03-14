#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
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
                                this->inputDims.push_back(800);
                                this->inputDims.push_back(1088);
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
        this->pred_ = outputTensorValues[0].GetTensorMutableData<float>();
        this->pred_dim = outputTensorValues[0].GetTensorTypeAndShapeInfo().GetShape();
        this->label_value_ = outputTensorValues[1].GetTensorMutableData<int64_t>();
        this->seg_ = outputTensorValues[2].GetTensorMutableData<float>();
        this->seg_dim_ = outputTensorValues[2].GetTensorTypeAndShapeInfo().GetShape();
        this->get_candidates();

}

void InstanceSegmentation::get_candidates(){
        this->instances_.clear();
        Instance ins;
        for(int batch = 0; batch < this->pred_dim[0]; batch++)
        {
                for(int cand = 0; cand < this->pred_dim[1]; cand++)
                {
                        int idx1 = cand * this->pred_dim[2];
                        int idx2 = idx1 + 4;
                        if(this->pred_[idx2] > this->bbox_thresh_)
                        {
                                int label_idx = idx1 / 5;
                                
                                ins.rect.x= this->pred_[idx1 + 0]; 
                                ins.rect.y = this->pred_[idx1 + 1];
                                ins.rect.width = this->pred_[idx1 + 2] - this->pred_[idx1 + 0];
                                ins.rect.height = this->pred_[idx1 + 3] - this->pred_[idx1 + 1];
                                ins.prob = this->pred_[idx1 + 4];
                                ins.label = this->label_value_[label_idx];
                                this->instances_.push_back(ins);
                        }

                }
        }

        
}


void InstanceSegmentation::DrawResult(cv::Mat &image){
        std::cout << this->instances_.size() << std::endl; 
        for(int i = 0 ; i < this->instances_.size() ; i++)
        {
                Instance ins = this->instances_[i];
                cv::rectangle(image, ins.rect , cv::Scalar(255,0,0) , 2);
        }
}
