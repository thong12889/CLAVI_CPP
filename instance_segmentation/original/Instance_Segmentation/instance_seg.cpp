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
#define BBOX_CONF_THRESH 0.5

static const int INPUT_W = 1333;
static const int INPUT_H = 800;

struct Instance
{
	cv::Rect_<float> rect;
	cv::Mat seg;
	int label;
	float prob;
};

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

static void get_candidates(const float* pred, std::vector<int64_t> pred_dim, const int64_t* label, std::vector<Instance>& instances)
{
	for(int batch = 0; batch < pred_dim[0]; batch++)
	{
		for(int cand = 0; cand < pred_dim[1]; cand++)
		{
			int score = 4;
			int idx1 = cand * pred_dim[2];
			int idx2 = idx1 + score;
			if(pred[idx2] > BBOX_CONF_THRESH)
			{
				int label_idx = idx1 / 5;
				Instance ins;
				ins.rect.x = pred[idx1 + 0];
				ins.rect.y = pred[idx1 + 1];
				ins.rect.width = pred[idx1 + 2];
				ins.rect.height = pred[idx1 + 3];
				ins.label = label[label_idx];
				ins.prob = pred[idx2];

				instances.push_back(ins);
			}
		}
	}
}

static void decode_outputs(const float* pred, std::vector<int64_t> pred_dim, const int64_t* label, const float* seg, std::vector<int64_t> seg_dim, std::vector<Instance>& instances, float scale, const int img_w, const int img_h)
{
	//cv::resize(result_img, result_img, cv::Size(img_w, img_h));
	get_candidates(pred, pred_dim, label, instances);

	//for(int i = 0; i < instances.size(); i++)
	//{
	//	float x0 = instances[i].rect.x / scale;
	//	float y0 = instances[i].rect.y / scale;
	//	float x1 = instances[i].rect.width / scale;
	//	float y1 = instances[i].rect.height / scale;
	//
	//	x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
	//	y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
	//	x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
	//	y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
	//
	//	instances[i].rect.x = x0;
	//	instances[i].rect.y = y0;
	//	instances[i].rect.width = x1 - x0;
	//	instances[i].rect.height = y1 - y0;
	//}
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

std::vector<cv::Vec3b> randColor(int num_classes, std::vector<cv::Vec3b>& listColor)
{
	for(int i = 0; i < num_classes; i++){
		int r, g, b;
		r = rand()%256;	
		g = rand()%256;	
		b = rand()%256;	
		listColor.push_back(cv::Vec3b(b,g,r));
	}
	
	return listColor;
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
	std::string imageFilepath = argv[2];
	std::string labelFilepath = argv[3];

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
		inputDims.push_back(800);
		inputDims.push_back(1333);
	}
	else
	{
		inputDims = getInputDims;
	}
	std::cout << inputDims << std::endl;

	const char* outputName0 = session.GetOutputName(0, allocator);
	const char* outputName1 = session.GetOutputName(1, allocator);
	const char* outputName2 = session.GetOutputName(2, allocator);

	std::vector<const char*> inputNames{inputName};
	std::vector<const char*> outputNames{outputName0, outputName1, outputName2};
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
		cv::dnn::blobFromImage(resizedImage, preprocessedImage, 1 / 127.5f, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), true, false);
		//cv::dnn::blobFromImage(resizedImage, preprocessedImage);

		inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());

		inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));

		auto outputTensorValues = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), numInputNodes, outputNames.data(), numOutputNodes);
		
		//Read image file
		cv::Mat image = imread_t(imageFilepath);
		int img_w = image.cols;
		int img_h = image.rows;
		float scale = std::min(INPUT_W / (image.cols * 1.0), INPUT_H / (image.rows * 1.0));
		//cv::Mat result_img;
		std::vector<Instance> instances;

		const float* pred = outputTensorValues[0].GetTensorMutableData<float>();
		std::vector<int64_t> pred_dim = outputTensorValues[0].GetTensorTypeAndShapeInfo().GetShape();
		const int64_t* label_value = outputTensorValues[1].GetTensorMutableData<int64_t>();
		const float* seg = outputTensorValues[2].GetTensorMutableData<float>();
		std::vector<int64_t> seg_dim = outputTensorValues[2].GetTensorTypeAndShapeInfo().GetShape();
	
		//Get results
		std::vector<cv::Vec3b> color_list;
		randColor(labels.size(), color_list);
		decode_outputs(pred, pred_dim, label_value, seg, seg_dim, instances, scale, img_w, img_h);
		std::cout << pred_dim << std::endl;
		std::cout << instances.size() << std::endl;
		for(int i = 0; i < instances.size(); i++)
		{
			const Instance ins = instances[i];

			cv::Scalar color = cv::Scalar(color_list[ins.label][0], color_list[ins.label][1], color_list[ins.label][2]);
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

			cv::rectangle(resizedImage, ins.rect, color, 2);
		}
		
		//const float opacity = 0.5;
		//image = image * (1 - opacity) + result_img * opacity;

		cv::imwrite("../result.jpg", resizedImage);	
		//cv::namedWindow("Result", cv::WINDOW_NORMAL|cv::WINDOW_FREERATIO);
		//cv::imshow("Result", image);
		//cv::waitKey(0);
		//cv::destroyWindow("Result");
				
		end = clock();
		double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
		std::cout << time_taken << std::endl;
	}
}	
