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


#define tcout  std::cout
#define file_name_t  std::string
#define imread_t  cv::imread
#define NMS_THRESH 0.5
#define BBOX_CONF_THRESH 0.3

static const int INPUT_W = 224;
static const int INPUT_H = 224;

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

static void get_results_pixel(const int64_t* pred, std::vector<int64_t> pred_dim, std::vector<cv::Vec3b> color, cv::Mat& mat)
{	
	mat = cv::Mat(cv::Size(pred_dim[3], pred_dim[2]), CV_8UC3);
	for(int batch = 0; batch < pred_dim[0]; batch++)
	{
		for(int h = 0; h < pred_dim[2]; h++)
		{
			for(int w = 0; w < pred_dim[3]; w++)
			{
				int idx = (h * pred_dim[3]) + w;

				if(pred[idx] == 0)
				{
					//mat.at<cv::Vec3b>(h, w)[0] = 0;
					//mat.at<cv::Vec3b>(h, w)[1] = 0;
					//mat.at<cv::Vec3b>(h, w)[2] = 0;
					mat.at<cv::Vec3b>(h, w)[0] = color[pred[idx]][0];	
					mat.at<cv::Vec3b>(h, w)[1] = color[pred[idx]][1];	
					mat.at<cv::Vec3b>(h, w)[2] = color[pred[idx]][2];	
				}
				else
				{
					mat.at<cv::Vec3b>(h, w)[0] = color[pred[idx]][0];	
					mat.at<cv::Vec3b>(h, w)[1] = color[pred[idx]][1];	
					mat.at<cv::Vec3b>(h, w)[2] = color[pred[idx]][2];	
				}
			}
		}
	}
}

static void decode_outputs(const int64_t* pred, std::vector<int64_t> pred_dim, cv::Mat& result_img, std::vector<cv::Vec3b> color, float scale, const int img_w, const int img_h)
{
	get_results_pixel(pred, pred_dim, color, result_img);
	cv::resize(result_img, result_img, cv::Size(img_w, img_h));
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

int main(int argc, char* argv[])
{
	clock_t start, end;

	if(argc < 5)
	{
		std::cerr << "Usage CPU: [apps] [path/to/model] [path/to/image] [path/to/labal] --use_cpu" << std::endl; 
		std::cerr <<  "Usage GPU: [apps] [path/to/model] [path/to/image] [path/to/label] --use_cuda" << std::endl;
		return EXIT_FAILURE;
	}

	bool useCUDA{false};
	const char* useCUDAFlag = "--use_cuda";
	const char* useCPUFlag = "--use_cpu";
	if(strcmp(argv[4], useCUDAFlag) == 0)
	{
		useCUDA = true;
	}
	else if(strcmp(argv[4], useCPUFlag) == 0)
	{
		useCUDA = false;
	}
	else
	{
		throw std::runtime_error("Too many arguments.");
	}

	std::string instanceName{"object-detection-inference"};
	std::string modelFilepath = argv[1];
	std::string labelFilepath = argv[2];
	std::string imageFilepath = argv[3];

	Ort::Env env;
	Ort::SessionOptions session_options;

	if(useCUDA)
	{
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
	else
	{
		std::cout << "Inference Execution Provider: CPU" << std::endl;
		session_options.SetIntraOpNumThreads(1);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	}
	
	Ort::Session session(env, modelFilepath.c_str(), session_options);
	Ort::AllocatorWithDefaultOptions allocator;

	size_t numInputNodes = session.GetInputCount();
	size_t numOutputNodes = session.GetOutputCount();

	const char* inputName = session.GetInputName(0, allocator);

	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);	
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

	std::vector<int64_t> getInputDims = inputTensorInfo.GetShape();
	std::vector<int64_t> inputDims;
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
	std::cout << inputDims << std::endl;

	const char* outputName0 = session.GetOutputName(0, allocator);

	std::vector<const char*> inputNames{inputName};
	std::vector<const char*> outputNames{outputName0};

	std::vector<Ort::Value> inputTensors;

	size_t inputTensorSize = vectorProduct(inputDims);
	std::vector<float> inputTensorValues(inputTensorSize);
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	
	std::vector<std::string> labels{readLabels(labelFilepath)};

	for(int n = 0; n < 1; n++){

		start = clock();

		cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
		cv::Mat preprocessedImage;

		cv::Mat resizedImage;
		cv::resize(imageBGR, resizedImage, cv::Size(INPUT_W, INPUT_H));
		cv::dnn::blobFromImage(resizedImage, preprocessedImage);

		inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());

		inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));

		auto outputTensorValues = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), numInputNodes, outputNames.data(), numOutputNodes);
		
		//Read image file
		cv::Mat image = imread_t(imageFilepath);
		int img_w = image.cols;
		int img_h = image.rows;
		float scale = std::min(INPUT_W / (image.cols * 1.0), INPUT_H / (image.rows * 1.0));
		cv::Mat result_img;

		const float* pred = outputTensorValues[0].GetTensorMutableData<float>();
		std::vector<int64_t> pred_dim = outputTensorValues[0].GetTensorTypeAndShapeInfo().GetShape();

		std::vector<float> pred_list;
		for(int i = 0; i < 5; i++)
		{
			std::cout << pred[i] << std::endl;
			pred_list.push_back(pred[i]);
		}
		std::vector<float>::iterator iter = std::max_element(pred_list.begin(), pred_list.end());
		size_t index = std::distance(pred_list.begin(), iter);
		std::cout << "Result: " << labels[index] << std::endl;
		std::cout << "Score : " << pred[index]<< std::endl;

		end = clock();
		double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
		std::cout << "Processing time: " << time_taken << std::endl;
	}
}	
