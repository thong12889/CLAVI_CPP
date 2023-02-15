#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>

#include <opencv2/core/core.hpp>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "CVQueue.h"

#include <thread>


#define tcout  std::cout
#define file_name_t  std::string
#define imread_t  cv::imread
#define NMS_THRESH 0.5
#define BBOX_CONF_THRESH 0.3

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 80;

std::vector<int64_t> inputDims;
std::vector<const char*> inputNames;
std::vector<const char*> outputNames;
std::vector<Ort::Value> inputTensors;
size_t numInputNodes;
size_t numOutputNodes;
Ort::Session * session_ptr;
std::string modelFilepath;
std::string labelFilepath;
std::string imageFilepath;

//Testing Performance 
std::ofstream myfile;
clock_t start, end;
//Testing Performance 

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
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


cv::Mat static_resize(cv::Mat& img)
{
	float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
	int unpad_w = r * img.cols;
	int unpad_h = r * img.rows;
	cv::Mat re(unpad_h, unpad_w, CV_8UC3);
	cv::resize(img, re, re.size());
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114,114,114));
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
	return out;
}

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
};

struct GridAndStride
{
	int grid0;
	int grid1;
	int stride;
};

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
	for(auto stride : strides)
	{
		int num_grid_w = target_w / stride;
		int num_grid_h = target_h / stride;
		for(int g1 = 0; g1 < num_grid_h; g1++)
		{
			for(int g0 = 0; g0 < num_grid_w; g0++)
			{
				grid_strides.push_back((GridAndStride){g0, g1, stride});
			}
		}
	}
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects)
{
	const int num_anchors = grid_strides.size(); std::cout <<"size grid_strides: " << grid_strides.size()<< std::endl;

	for(int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
	{
		const int grid0 = grid_strides[anchor_idx].grid0;
		const int grid1 = grid_strides[anchor_idx].grid1;
		const int stride = grid_strides[anchor_idx].stride;

		const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

		float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
		float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
		float w = exp(feat_ptr[basic_pos + 2]) * stride;
		float h = exp(feat_ptr[basic_pos + 3]) * stride;

		float x0 = x_center - w * 0.5f;
		float y0 = y_center - h * 0.5f;
		
		float box_objectness = feat_ptr[basic_pos + 4];

		for(int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
		{
			float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
			float box_prob = box_objectness * box_cls_score;
			if(box_prob > prob_threshold)
			{
				Object obj;
				obj.rect.x = x0;
				obj.rect.y = y0;
				obj.rect.width = w;
				obj.rect.height = h;
				obj.label = class_idx;
				obj.prob = box_prob;

				objects.push_back(obj);
			}
		}
	}
}

static inline float intersection_area(const Object& a, const Object& b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while(i <= j)
	{
		while(faceobjects[i].prob > p)
			i++;

		while(faceobjects[j].prob < p)
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

static void qsort_descent_inplace(std::vector<Object>& objects)
{
	if(objects.empty())
		return;

	qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
	picked.clear();

	const int n = faceobjects.size();

	std::vector<float> areas(n);
	for(int i = 0; i < n; i++)
	{
		areas[i] = faceobjects[i].rect.area();
	}

	for(int i = 0; i < n; i++)
	{
		const Object& a = faceobjects[i];

		int keep = 1;
		for(int j = 0; j < (int)picked.size(); j++)
		{
			const Object& b = faceobjects[picked[j]];

			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			if(inter_area / union_area > nms_threshold)
			{
				keep = 0;
			}
		}

		if(keep)
			picked.push_back(i);

	}
}

static void get_candidates(const float* pred, std::vector<long> pred_dim, const int64_t* label, std::vector<Object>& objects)
{
	for(int batch = 0; batch < pred_dim[0]; batch++)
	{
		for(int cand = 0; cand < pred_dim[1]; cand++)
		{
			int score = 4;
			int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
			int idx2 = idx1 + score;
			if(pred[idx2] > BBOX_CONF_THRESH)
			{
				int label_idx = idx1 / 5;
				Object obj;
				obj.rect.x = pred[idx1 + 0];
				obj.rect.y = pred[idx1 + 1];
				obj.rect.width = pred[idx1 + 2];
				obj.rect.height = pred[idx1 + 3];
				obj.label = label[label_idx];
				obj.prob = pred[idx2];
				
				objects.push_back(obj);
			}
		}
	}
}

static void decode_outputs(const float* pred, std::vector<long> pred_dim, const int64_t* label, std::vector<Object>& objects, float scale, const int img_w, const int img_h)
{
	get_candidates(pred, pred_dim, label, objects);

	float dif_w = (float)img_w / (float)INPUT_W;
	float dif_h = (float)img_h / (float)INPUT_H;
	for(int i = 0; i < objects.size(); i++)
	{
		float x0 = objects[i].rect.x / scale;
		float y0 = objects[i].rect.y / scale;
		float x1 = (objects[i].rect.width) / scale;
		float y1 = (objects[i].rect.height) / scale;

		x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;
	}
}

const float color_list[80][3] = 
{
	{0.000, 0.447, 0.741},
	{0.850, 0.325, 0.098},
    	{0.929, 0.694, 0.125},
    	{0.494, 0.184, 0.556},
    	{0.466, 0.674, 0.188},
    	{0.301, 0.745, 0.933},
    	{0.635, 0.078, 0.184},
    	{0.300, 0.300, 0.300},
    	{0.600, 0.600, 0.600},
    	{1.000, 0.000, 0.000},
    	{1.000, 0.500, 0.000},
    	{0.749, 0.749, 0.000},
    	{0.000, 1.000, 0.000},
    	{0.000, 0.000, 1.000},
    	{0.667, 0.000, 1.000},
    	{0.333, 0.333, 0.000},
    	{0.333, 0.667, 0.000},
    	{0.333, 1.000, 0.000},
    	{0.667, 0.333, 0.000},
    	{0.667, 0.667, 0.000},
    	{0.667, 1.000, 0.000},
    	{1.000, 0.333, 0.000},
    	{1.000, 0.667, 0.000},
    	{1.000, 1.000, 0.000},
    	{0.000, 0.333, 0.500},
    	{0.000, 0.667, 0.500},
    	{0.000, 1.000, 0.500},
    	{0.333, 0.000, 0.500},
    	{0.333, 0.333, 0.500},
    	{0.333, 0.667, 0.500},
    	{0.333, 1.000, 0.500},
    	{0.667, 0.000, 0.500},
    	{0.667, 0.333, 0.500},
    	{0.667, 0.667, 0.500},
    	{0.667, 1.000, 0.500},
    	{1.000, 0.000, 0.500},
    	{1.000, 0.333, 0.500},
    	{1.000, 0.667, 0.500},
    	{1.000, 1.000, 0.500},
    	{0.000, 0.333, 1.000},
   	{0.000, 0.667, 1.000},
    	{0.000, 1.000, 1.000},
    	{0.333, 0.000, 1.000},
    	{0.333, 0.333, 1.000},
    	{0.333, 0.667, 1.000},
    	{0.333, 1.000, 1.000},
    	{0.667, 0.000, 1.000},
    	{0.667, 0.333, 1.000},
    	{0.667, 0.667, 1.000},
    	{0.667, 1.000, 1.000},
    	{1.000, 0.000, 1.000},
    	{1.000, 0.333, 1.000},
    	{1.000, 0.667, 1.000},
    	{0.333, 0.000, 0.000},
    	{0.500, 0.000, 0.000},
    	{0.667, 0.000, 0.000},
    	{0.833, 0.000, 0.000},
    	{1.000, 0.000, 0.000},
    	{0.000, 0.167, 0.000},
    	{0.000, 0.333, 0.000},
    	{0.000, 0.500, 0.000},
    	{0.000, 0.667, 0.000},
    	{0.000, 0.833, 0.000},
    	{0.000, 1.000, 0.000},
    	{0.000, 0.000, 0.167},
    	{0.000, 0.000, 0.333},
    	{0.000, 0.000, 0.500},
    	{0.000, 0.000, 0.667},
   	{0.000, 0.000, 0.833},
    	{0.000, 0.000, 1.000},
    	{0.000, 0.000, 0.000},
    	{0.143, 0.143, 0.143},
    	{0.286, 0.286, 0.286},
    	{0.429, 0.429, 0.429},
    	{0.571, 0.571, 0.571},
    	{0.714, 0.714, 0.714},
    	{0.857, 0.857, 0.857},
   	{0.000, 0.447, 0.741},
    	{0.314, 0.717, 0.741},
    	{0.50, 0.5, 0}
};

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	static const char* class_names[] = {
		"sunglasses",
		"hat",
   		"jacket",
    		"shirt",
    		"pants",
    		"shorts",
    		"skirt",
    		"dress",
    		"bag",
    		"shoe",
    		"top"
	};

	cv::Mat image = bgr.clone();

	for(size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
		float c_mean = cv::mean(color)[0];
		cv::Scalar txt_color;
		if(c_mean > 0.5)
		{
			txt_color = cv::Scalar(0,0,0);
		}
		else
		{
			txt_color = cv::Scalar(255,255,255);
		}

		cv::rectangle(image, obj.rect, color * 255, 2);

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

		cv::Scalar txt_bk_color = color * 0.7 * 255;

		int x = obj.rect.x;
		int y = obj.rect.y + 1;
		if(y > image.rows)
			y = image.rows;

		cv::rectangle(image, cv::Rect(cv::Point(x,y), cv::Size(label_size.width, label_size.height + baseLine)), txt_bk_color, -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
	}

	cv::imwrite("_demo.jpg", image);
	fprintf(stderr, "save vis file\n");
	
	//cv::imshow("DEMO", image);
	//cv::waitKey(0);
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

std::vector<std::string> readLabels(std::string& labelFilepath)
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

void inference_thread(){
	cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

	cv::Mat preprocessedImage, resizedImage;
	std::vector<std::string> labels{readLabels(labelFilepath)};
	if(true)
	{
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

		for(int i = 0; i < 1000; i++)
		{
			
			// cv::Mat resizedImage = static_resize(imageBGR);
			cv::resize(imageBGR, resizedImage , cv::Size(640,640));
			cv::dnn::blobFromImage(resizedImage, preprocessedImage);

			
		
			size_t inputTensorSize = vectorProduct(inputDims);
			std::vector<float> inputTensorValues(inputTensorSize);

			start = clock();
			inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
			

			inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
			
			auto outputTensorValues = (*session_ptr).Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), numInputNodes, outputNames.data(), numOutputNodes);

			
			
			
			//Read image file
			// cv::Mat image = imread_t(imageFilepath);
			int img_w = imageBGR.cols;
			int img_h = imageBGR.rows;
			float scale = std::min(INPUT_W / (imageBGR.cols * 1.0), INPUT_H / (imageBGR.rows * 1.0));
			std::vector<Object> objects;

			const float* pred = outputTensorValues[0].GetTensorMutableData<float>();
			std::vector<long> pred_dim = outputTensorValues[0].GetTensorTypeAndShapeInfo().GetShape();
			const int64_t* label = outputTensorValues[1].GetTensorMutableData<int64_t>();
			std::vector<long> label_dim = outputTensorValues[1].GetTensorTypeAndShapeInfo().GetShape();
			
			

			//Get results
			decode_outputs(pred, pred_dim, label, objects, scale, img_w, img_h);

			end = clock();
			myfile << std::to_string(double(end- start) / double(CLOCKS_PER_SEC));
			myfile << "\n";
			//Read label file
			
			

			//cv::imwrite("tech_result.jpg", image);
		}	
	}

	//ReleaseCUDAProviderOptions(cuda_options);
	cv::destroyAllWindows();
}

int main(int argc, char* argv[])
{
	
	myfile.open("performance_testing/obj_model.csv");
	myfile << "Object Detection\n";
	

	if(argc < 4)
	{
		std::cerr << "Usage CPU: [apps] [path/to/model] [path/to/image] [path/to/labal]" << std::endl; 
		std::cerr <<  "Usage GPU: [apps] [path/to/model] [path/to/image] [path/to/label] --use_cuda" << std::endl;
		return EXIT_FAILURE;
	}

	bool Camera{true};
	bool Onetime{true};
	bool useCUDA{true};
	const char* useCUDAFlag = "--use_cuda";
	const char* useCPUFlag = "--use_cpu";
	// if(argc == 4)
	// {
	// 	useCUDA = false;
	// }
	// else if ((argc == 5) && (strcmp(argv[1], useCUDAFlag) == 0))
	// {
	// 	useCUDA = true;
	// }
	// else if ((argc == 5) && (strcmp(argv[1], useCPUFlag) == 0))
	// {
	// 	useCUDA = false;
	// }
	// else if ((argc == 5) && (strcmp(argv[1], useCUDAFlag) != 0))
	// {
	// 	useCUDA = false;
	// }
	// else
	// {
	// 	throw std::runtime_error("Too many arguments.");
	// }

	useCUDA = true;
	if(useCUDA)
	{
		std::cout << "Inference Execution Provider: CUDA" << std::endl;
	}
	else
	{
		std::cout << "Inference Execution Provider: CPU" << std::endl;
	}

	std::string instanceName{"object-detection-inference"};
	modelFilepath = argv[1];
	labelFilepath = argv[2];
	imageFilepath = argv[3];

	Ort::Env env;
	Ort::SessionOptions session_options;
	Ort::AllocatorWithDefaultOptions allocator;
	if(!useCUDA){
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = 0;
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch();
		cuda_options.cuda_mem_limit = static_cast<int>(SIZE_MAX * 1024 * 1024);
		cuda_options.arena_extend_strategy = 1;
		cuda_options.do_copy_in_default_stream = 1;
		session_options.AppendExecutionProvider_CUDA(cuda_options);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	}
	else{
		//CUDA
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = 0;
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch();
		cuda_options.cuda_mem_limit = static_cast<int>(SIZE_MAX * 1024 * 1024);
		cuda_options.arena_extend_strategy = 1;
		cuda_options.do_copy_in_default_stream = 1;
		session_options.AppendExecutionProvider_CUDA(cuda_options);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	}

	Ort::Session session(env, modelFilepath.c_str(), session_options);

	numInputNodes = session.GetInputCount();
	numOutputNodes = session.GetOutputCount();

	const char* inputName = session.GetInputName(0, allocator);

	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);	
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

	inputDims = inputTensorInfo.GetShape();

	const char* outputName0 = session.GetOutputName(0, allocator);
	const char* outputName1 = session.GetOutputName(1, allocator);

	inputNames.push_back(inputName);
	outputNames.push_back(outputName0);
	outputNames.push_back(outputName1);

	session_ptr = &session;

	std::thread display_thread(inference_thread);
	display_thread.join();

	
	return 0;
}	
