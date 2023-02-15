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


ObjectClassification *cls;

//Write to CSV
std::ofstream myfile;


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
	myfile.open("performance_testing/cls_model.csv");
	myfile << "Object Classification\n";
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

		start = clock();

		cls->Inference(preprocessedImage);

		//#####Perfomance Testing######
		end = clock();
		double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
		myfile << std::to_string(time_taken);
		myfile << "\n";
		std::cout << "Processing time: " << time_taken << std::endl;
		//#####Perfomance Testing######

		cls->DrawResult(resizedImage);

		cv::resize(resizedImage, resizedImage, cv::Size(500, 500));

		cv::imshow("Test" , resizedImage);
		if(cv::waitKey(30) == 'q'){
			break;
		}


		

		
	}
	myfile.close();
}	
