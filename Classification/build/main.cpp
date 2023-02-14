#include <iostream>
#include "ObjectClassification.h"
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>

#include <iterator>
#include <memory>
#include <vector>


ObjectClassification *cls;


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

    cls = new ObjectClassification(modelFilepath, labelFilepath);

	Ort::Env env;
	Ort::SessionOptions session_options;

	if(useCUDA)
	{
        //Use CUDA
        cls->UseCUDA(session_options);
	}
	else
	{
        //use CPU
        cls->UseCPU(session_options);

	}
	
	// Ort::Session session(env, modelFilepath.c_str(), session_options);
	// Ort::AllocatorWithDefaultOptions allocator;

	size_t numInputNodes;
	size_t numOutputNodes;

	const char* inputName;


	ONNXTensorElementDataType inputType;

	
	std::vector<int64_t> inputDims;

	const char* outputName0;

	std::vector<const char*> inputNames;
	std::vector<const char*> outputNames;
	std::vector<Ort::Value> inputTensors;

	size_t inputTensorSize ;
	std::vector<float> inputTensorValues;
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::Session session = cls->SessionInit(env,
                                session_options,
                                numInputNodes,
                                numOutputNodes,
                                inputDims,
                                inputNames,
                                outputNames,
                                inputTensors,
                                inputTensorSize,
                                inputTensorValues);


	
	std::vector<std::string> labels{readLabels(labelFilepath)};

	for(int n = 0; n < 1; n++){

		start = clock();

		cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
		cv::Mat preprocessedImage;

		cv::Mat resizedImage;
		cv::resize(imageBGR, resizedImage, cv::Size(224, 224));
		cv::dnn::blobFromImage(resizedImage, preprocessedImage);

		inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());

		inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));

		auto outputTensorValues = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), numInputNodes, outputNames.data(), numOutputNodes);
		
		//Read image file
		cv::Mat image = imread_t(imageFilepath);
		int img_w = image.cols;
		int img_h = image.rows;
		float scale = std::min(224 / (image.cols * 1.0), 224 / (image.rows * 1.0));
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
