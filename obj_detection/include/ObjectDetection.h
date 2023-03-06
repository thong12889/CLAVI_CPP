#include<iostream>
#include <vector>

//ONNX RUNTIME LIB
#include <onnxruntime_cxx_api.h>

//INTERNAL LIB
#include "GridAndStride.h"
#include "Object.h"

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




class ObjectDetection{
	public:
		ObjectDetection(std::string, std::string);
		std::vector<std::string> FindIndexClass(std::string&);
		void UseCUDA();
		void UseCPU();
		Ort::Session SessionInit();
		void InferenceInit(cv::Mat&);
		void RunInference(cv::Mat&);
		void get_candidates(const float* pred, std::vector<long> pred_dim, const int64_t* label);
		void decode_outputs(const float* pred, std::vector<long> pred_dim, const int64_t* label , float scale, const int img_w, const int img_h);
		const int GetInputH() const;
		const int GetInputW() const;
		void DrawResult(cv::Mat&);
		


		
	private:
		std::string modelFilepath;
		const float nms_thresh_ = 0.5;
		const float bb_conf_thresh_ = 0.7;
		const int input_w_ = 640;
		const int input_h_ = 640;
		std::vector<std::string> classes_label_;
		int num_classes_;
		Ort::Env env;
		Ort::SessionOptions session_options;
		Ort::AllocatorWithDefaultOptions allocator;
		const char* inputName;

		Ort::Session *session_ptr_;
		std::vector<Ort::Value> inputTensors;

		std::vector<int64_t> inputDims;
		std::vector<const char*> inputNames;
		std::vector<const char*> outputNames;
		size_t numInputNodes;
		size_t numOutputNodes;

		size_t inputTensorSize;
		std::vector<float> inputTensorValues;
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		std::vector<Ort::Value> outputTensorValues;

		const float* pred;
		std::vector<long> pred_dim;
		const int64_t* label;
		std::vector<long> label_dim;

		//Variable of Decode
		int img_w;
		int img_h;
		float scale;
		std::vector<Object> objects;
		


		
};


