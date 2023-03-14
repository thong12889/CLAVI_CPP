#include <iostream>
#include "InstanceSegmentation.h"
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

#include "include/plot/PerformancePlot.h"

// #define tcout  std::cout
#define file_name_t  std::string
#define imread_t  cv::imread
#define NMS_THRESH 0.5
#define BBOX_CONF_THRESH 0.3

static const int INPUT_W = 1088;
static const int INPUT_H = 800;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}



int main(int argc, char* argv[])
{
	InstanceSegmentation *instance;
	PerformancePlot *perf = new FPS("../log/" , "Instance Segmentation FPS Testing");

	if(argc < 5)
	{
		std::cerr << "Usage CPU: [apps] [path/to/model] [path/to/image] [path/to/labal] --use_cpu" << std::endl; 
		std::cerr <<  "Usage GPU: [apps] [path/to/model] [path/to/image] [path/to/label] --use_cuda" << std::endl;
		return EXIT_FAILURE;
	}

	const char* useCUDAFlag = "--use_gpu";
	const char* useCPUFlag = "--use_cpu";

	std::string instanceName{"object-detection-inference"};
	std::string modelFilepath = argv[1];
	std::string labelFilepath = argv[2];
	std::string imageFilepath = argv[3];

	instance = new InstanceSegmentation(modelFilepath, labelFilepath);

	if(strcmp(argv[4] , useCUDAFlag) == 0)
	{
		std::cout << "Use CUDA" << std::endl;
		//Use CUDA
		instance->UseCUDA();
	}
	else
	{
		std::cout << "USE CPU" << std::endl;
		//use CPU
		instance->UseCPU();
	}
	
	Ort::Session session = instance->SessionInit();
	cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
	cv::Mat preprocessedImage, resizedImage , img_display;

	for(int n = 0; n < 1000; n++){
		img_display = imageBGR.clone();

		cv::resize(imageBGR, resizedImage, cv::Size(1088, 800));
		resizedImage.convertTo(resizedImage, CV_32F, 1 / 255.0);

		cv::dnn::blobFromImage(resizedImage, preprocessedImage);

		perf->StartRecord();		
		instance->Inference(preprocessedImage);
		perf->Stamp();

		cv::resize(img_display , img_display  , cv::Size(1088, 800));
		instance->DrawResult(img_display);


		cv::imshow("Test" , img_display);
		if(cv::waitKey(30) == 'q'){
			break;
		}


		

		
	}
	perf->Show();
	perf->Save();
	perf->End();
}	
