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
        this->num_classes_ = this->FindIndexClass(labelFilepath);
        std::cout << "Total Class : " << std::to_string(num_classes_.size()) << std::endl;
}

std::vector<std::string> ObjectDetection::FindIndexClass(std::string &path){
        std::vector<std::string> path_temp;
        std::ifstream input(path);
        for( std::string line; getline( input, line ); )
        {
                path_temp.push_back(line);
        }
        return path_temp;
        
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
        std::cout << "PASS1" << std::endl;
        this->inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
        this->inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, this->inputTensorValues.data(), this->inputTensorSize, this->inputDims.data(), this->inputDims.size()));
        this->outputTensorValues = (*session_ptr_).Run(Ort::RunOptions{nullptr}, this->inputNames.data(), this->inputTensors.data(), this->numInputNodes, this->outputNames.data(), this->numOutputNodes);
        std::cout << "PASS2" << std::endl;
        this->pred = this->outputTensorValues[0].GetTensorMutableData<float>();
        this->pred_dim = this->outputTensorValues[0].GetTensorTypeAndShapeInfo().GetShape();
        this->label = this->outputTensorValues[1].GetTensorMutableData<int64_t>();
        this->label_dim = this->outputTensorValues[1].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "PASS3" << std::endl;
        decode_outputs(this->pred, this->pred_dim, this->label, this->objects, this->scale, this->img_w, this->img_h);
        std::cout << "PASS4" << std::endl;


}


template <typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v){

	os << "[";
        for(int i=0; i < v.size(); ++i)
        {
                os << v[i];
                if(i != v.size() - 1)
                {
                        os << ", ";
                }
        }
        os << "]";
        return os;
}




void ObjectDetection::generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides){
	GridAndStride *gas = new GridAndStride();
	for(auto stride : strides)
        {
                int num_grid_w = target_w / stride;
                int num_grid_h = target_h / stride;
                for(int g1 = 0; g1 < num_grid_h; g1++)
                {
                        for(int g0 = 0; g0 < num_grid_w; g0++)
                        {
				gas->grid_0_ = g0;
				gas->grid_1_ = g1;
				gas->stride_ = stride;
                                grid_strides.push_back(*gas);
                        }
                }
        }

}


inline float ObjectDetection::intersection_area(const Object& a, const Object& b){
	cv::Rect_<float> inter = a.rect_ & b.rect_;
        return inter.area();
}

void ObjectDetection::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right){
	int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob_;

        while(i <= j)
        {
                while(faceobjects[i].prob_ > p)
                        i++;

                while(faceobjects[j].prob_ < p)
                        j--;

                if(i <= j)
                {
                        std::swap(faceobjects[i], faceobjects[j]);

                        i++;
                        j--;
                }
        }

        #pragma omp parallel sections
        {
                #pragma omp section
                {
                        if(left < j) qsort_descent_inplace(faceobjects, left, j);
                }
                #pragma omp section
                {
                        if(i < right) qsort_descent_inplace(faceobjects, i, right);
                }
        }
}

void ObjectDetection::qsort_descent_inplace(std::vector<Object>& objects){
	if(objects.empty())
                return;

        qsort_descent_inplace(objects, 0, objects.size() - 1);	
}

void ObjectDetection::nms_sorted_bboxes(const std::vector<Object>& object){
        

}

void ObjectDetection::get_candidates(const float* pred, std::vector<long> pred_dim, const int64_t* label, std::vector<Object>& objects){
        objects.clear();
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

                                objects.push_back(obj);
                        }
                }
        }
}

void ObjectDetection::decode_outputs(const float* pred, std::vector<long> pred_dim, const int64_t* label, std::vector<Object>& objects, float scale, const int img_w, const int img_h){
	get_candidates(pred, pred_dim, label, objects);

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


std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type)
{
	switch(type)
	{
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
			os << "undefined";
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			os << "float";
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			os << "uint8_t";
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			os << "int8_t";
			break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            		os << "uint16_t";
           	 	break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            		os << "int16_t";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            		os << "int32_t";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            		os << "int64_t";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            		os << "std::string";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            		os << "bool";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            		os << "float16";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            		os << "double";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            		os << "uint32_t";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            		os << "uint64_t";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            		os << "float real + float imaginary";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            		os << "double real + float imaginary";
            		break;
        	case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            		os << "bfloat16";
            		break;
        	default:
            		break;
	}
	return os;
}

std::vector<std::string> ObjectDetection::readLabels(std::string& labelFilepath)
{
	std::vector<std::string> labels;
	std::string line;
	std::ifstream fp(labelFilepath);
	while(std::getline(fp, line))
	{
		labels.push_back(line);
	}
	return labels;
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
                // cv::putText(frame, std::to_string(obj.label_), cv::Point(obj.rect_.x , obj.rect_.y) , cv::FONT_HERSHEY_SIMPLEX, 0.4,1, 0 );
	
		// cv::Size label_size = cv::getTextSize(labels[obj.label_], cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, 0);
	
		// cv::Scalar txt_bk_color = color * 0.7 * 255;
	
		// int x = obj.rect_.x;
		// int y = obj.rect_.y + 1;
		// if(y > frame.rows)
		// 	y = frame.rows;
		
		// // //Draw label
		// cv::rectangle(frame, cv::Rect(cv::Point(x,y), cv::Size(label_size.width, label_size.height + 0)), txt_bk_color, -1);
		// cv::putText(frame, labels[obj.label_], cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
	}

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
