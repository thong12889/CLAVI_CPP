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
		template <typename T> friend std::ostream& operator<<(std::ostream& , const std::vector<T>& );
		void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
		void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects);
		inline float intersection_area(const Object& a, const Object& b);
		void qsort_descent_inplace(std::vector<Object>& , int left, int right);
		void qsort_descent_inplace(std::vector<Object>& );
		void nms_sorted_bboxes(const std::vector<Object>& );
		void get_candidates(const float* pred, std::vector<long> pred_dim, const int64_t* label, std::vector<Object>& );
		void decode_outputs(const float* pred, std::vector<long> pred_dim, const int64_t* label, std::vector<Object>& , float scale, const int img_w, const int img_h);
		void draw_objects(const cv::Mat& bgr, const std::vector<Object>& );
		friend std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type);
		std::vector<std::string> readLabels(std::string& labelFilepath);
		const int GetInputH() const;
		const int GetInputW() const;
		void DrawResult(cv::Mat&);
		


		
	private:
		std::string modelFilepath;
		const float nms_thresh_ = 0.5;
		const float bb_conf_thresh_ = 0.7;
		const int input_w_ = 640;
		const int input_h_ = 640;
		std::vector<std::string> num_classes_;
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


