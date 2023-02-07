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
#include <iostream>
#include <limits>
#include <numeric>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <memory>




class ObjectDetection{
	public:
		ObjectDetection();
		Ort::Session SessionInitCLAVI(std::string, std::vector<const char*>&, std::vector<const char*>&, std::vector<int64_t>&, size_t& , size_t& );
		template <typename T> friend std::ostream& operator<<(std::ostream& , const std::vector<T>& );
		cv::Mat StaticResize(cv::Mat &);
		void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
		void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects);
		inline float intersection_area(const Object& a, const Object& b);
		void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
		void qsort_descent_inplace(std::vector<Object>& objects);
		void nms_sorted_bboxes(const std::vector<Object>& faceobjects);
		void get_candidates(const float* pred, std::vector<long> pred_dim, const int64_t* label, std::vector<Object>& objects);
		void decode_outputs(const float* pred, std::vector<long> pred_dim, const int64_t* label, std::vector<Object>& objects, float scale, const int img_w, const int img_h);
		void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);
		friend std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type);
		std::vector<std::string> readLabels(std::string& labelFilepath);
		const int GetInputH() const;
		const int GetInputW() const;
		void DrawResult(std::vector<Object>& , cv::Mat&, std::vector<std::string>);
		void UseCUDA();
		void UseCPU();


		
	private:
		const float nms_thresh_ = 0.9;
		const float bb_conf_thresh_ = 0.3;
		const int input_w_ = 640;
		const int input_h_ = 640;
		int num_classes_ = 80;
		Ort::Env env;
		Ort::SessionOptions session_options;
		Ort::AllocatorWithDefaultOptions allocator;
		const char* inputName;

		std::vector<Ort::Value> inputTensors;

		std::vector<int64_t> inputDims;
		std::vector<const char*> inputNames;
		std::vector<const char*> outputNames;
		size_t numInputNodes;
		size_t numOutputNodes;


		
};


