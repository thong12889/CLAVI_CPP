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

#include <fstream>

//Write to CSV
#include "include/plot/PerformancePlot.h"


// #define tcout  std::cout
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



int main(int argc, char* argv[])
{
	ObjectClassification *cls;

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



	if(useCUDA)
	{
        //Use CUDA
        cls->UseCUDA();
	}
	else
	{
        //use CPU
        cls->UseCPU();

	}
	


    Ort::Session session = cls->SessionInit();
	cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

	for(int n = 0; n < 100; n++){

		

		cv::Mat preprocessedImage, resizedImage;

		cv::resize(imageBGR, resizedImage, cv::Size(224, 224));
		cv::dnn::blobFromImage(resizedImage, preprocessedImage);


		cls->Inference(preprocessedImage);


		cls->DrawResult(resizedImage);

		cv::resize(resizedImage, resizedImage, cv::Size(500, 500));

		cv::imshow("Test" , resizedImage);
		if(cv::waitKey(30) == 'q'){
			break;
		}


		

		
	}
}	
