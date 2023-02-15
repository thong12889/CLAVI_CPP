//miscellaneous LIB
#include <iostream>
#include <fstream>
#include <vector>


//OPENCV LIB
#include "opencv2/opencv.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>

//ONNX RUNTIME LIB
#include <onnxruntime_cxx_api.h>

//External LIB
#include "ObjectDetection.h"


ObjectDetection::ObjectDetection(std::string modelFilepath, std::string labelFilepath){
        this->modelFilepath = modelFilepath;
        this->classes_label_ = this->FindIndexClass(labelFilepath);
        this->num_classes_ = classes_label_.size();
}

std::vector<std::string> ObjectDetection::FindIndexClass(std::string &path){
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

Ort::Session ObjectDetection::SessionInit(){
        
        Ort::Session session(env, this->modelFilepath.c_str(), session_options);


        //OOP


        this->numInputNodes = session.GetInputCount();
	this->numOutputNodes = session.GetOutputCount();
        this->inputName = session.GetInputName(0, allocator);

	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	Ort::Unowned<Ort::TensorTypeAndShapeInfo>  inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

	this->inputDims = inputTensorInfo.GetShape();

	const char* outputName0 = session.GetOutputName(0, allocator);
	const char* outputName1 = session.GetOutputName(1, allocator);

        this->inputNames.push_back(inputName);
        this->outputNames.push_back(outputName0);
        this->outputNames.push_back(outputName1);

        session_ptr_ = &session;

        return session;
	

}

void ObjectDetection::UseCPU(){
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

void ObjectDetection::UseCUDA(){
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch();
        cuda_options.cuda_mem_limit = static_cast<int>(SIZE_MAX * 1024 * 1024);
        cuda_options.arena_extend_strategy = 1;
        cuda_options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}


template <typename T> T vectorProduct(std::vector<T>& v){
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

void ObjectDetection::InferenceInit(cv::Mat& frame_init){
        this->inputTensorSize = vectorProduct(this->inputDims);
        this->inputTensorValues.push_back(this->inputTensorSize);

        this->img_w = frame_init.cols;
        this->img_h = frame_init.rows;
        this->scale = std::min(this->input_w_ / (frame_init.cols * 1.0), this->input_h_ / (frame_init.rows * 1.0));
}

void ObjectDetection::RunInference(cv::Mat &preprocessedImage){

        this->inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
        this->inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, this->inputTensorValues.data(), this->inputTensorSize, this->inputDims.data(), this->inputDims.size()));
        this->outputTensorValues = (*session_ptr_).Run(Ort::RunOptions{nullptr}, this->inputNames.data(), this->inputTensors.data(), this->numInputNodes, this->outputNames.data(), this->numOutputNodes);

        this->pred = this->outputTensorValues[0].GetTensorMutableData<float>();
        this->pred_dim = this->outputTensorValues[0].GetTensorTypeAndShapeInfo().GetShape();
        this->label = this->outputTensorValues[1].GetTensorMutableData<int64_t>();
        this->label_dim = this->outputTensorValues[1].GetTensorTypeAndShapeInfo().GetShape();

        decode_outputs(this->pred, this->pred_dim, this->label, this->scale, this->img_w, this->img_h);



}


void ObjectDetection::get_candidates(const float* pred, std::vector<long> pred_dim, const int64_t* label){
        this->objects.clear();
	for(int batch = 0; batch < pred_dim[0]; batch++)
        {
                for(int cand = 0; cand < pred_dim[1]; cand++)
                {
                        int score = 4;
                        int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                        int idx2 = idx1 + score;
                        if(pred[idx2] > this->bb_conf_thresh_)
                        {
                                std::cout << "bb_thresh : " << std::to_string(this->bb_conf_thresh_) << std::endl;
                                int label_idx = idx1 / 5;
                                Object obj;
                                obj.rect_.x = pred[idx1 + 0];
                                obj.rect_.y = pred[idx1 + 1];
                                obj.rect_.width = pred[idx1 + 2];
                                obj.rect_.height = pred[idx1 + 3];
                                obj.label_ = label[label_idx];
                                obj.prob_ = pred[idx2];

                                this->objects.push_back(obj);
                        }
                }
        }
}

void ObjectDetection::decode_outputs(const float* pred, std::vector<long> pred_dim, const int64_t* label, float scale, const int img_w, const int img_h){
	get_candidates(pred, pred_dim, label);

        float dif_w = (float)img_w / (float)input_w_;
        float dif_h = (float)img_h / (float)input_h_;
        for(int i = 0; i < objects.size(); i++)
        {
                float x0 = objects[i].rect_.x / scale;
                float y0 = objects[i].rect_.y / scale;
                float x1 = (objects[i].rect_.width) / scale;
                float y1 = (objects[i].rect_.height) / scale;

                x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
                y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
                x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

                objects[i].rect_.x = x0;
                objects[i].rect_.y = y0;
                objects[i].rect_.width = x1 - x0;
                objects[i].rect_.height = y1 - y0;
        }
}


const int ObjectDetection::GetInputH() const{
        return this->input_h_;
}

const int ObjectDetection::GetInputW() const{
        return this->input_w_;
}

void ObjectDetection::DrawResult(cv::Mat& frame){
	//Draw results
        // std::cout << "Size : " << std::to_string(this->objects.size()) << std::endl;
	for(int i = 0; i < this->objects.size(); i++)
	{
                // std::cout<< "i : " << i << std::endl;
		const Object& obj = this->objects[i];
	
		cv::Scalar color = cv::Scalar(0, 255, 0);
		float c_mean = cv::mean(color)[0];
		cv::Scalar txt_color;
		if(c_mean > 0.5){
			txt_color = cv::Scalar(0,0,0);
		}
		else{
			txt_color = cv::Scalar(255,255,255);
		}
	
		// //Draw bounding box
		cv::rectangle(frame, obj.rect_, cv::Scalar(0,0,255), 2);
                cv::putText(frame, classes_label_[obj.label_], cv::Point(obj.rect_.x , obj.rect_.y) , cv::FONT_HERSHEY_SIMPLEX, 1,cv::Scalar(0,255,0) , true );
	}

}

