#include <iostream>
#include "ObjectDetection.h"
#include <vector>

#include <iterator>
#include <memory>
#include <string>

#include "CVQueue.h"

#include <gst/gst.h>
#include <gst/app/app.h>


std::chrono::time_point<std::chrono::system_clock> start, end;

// template <typename T> T vectorProduct(std::vector<T>& v){
// 	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
// }


int main(int argc, char* argv[]){
	

	if(argc < 4)
	{
		std::cerr << "Usage CPU: [apps] [path/to/model] [path/to/image] [path/to/labal]" << std::endl; 
		std::cerr <<  "Usage GPU: [apps] [path/to/model] [path/to/image] [path/to/label] --use_cuda" << std::endl;
		return EXIT_FAILURE;
	}

	bool Camera = true;
	bool Onetime = false;
	bool useCUDA = false;
	const char* useCUDAFlag = "--use_cuda";
	const char* useCPUFlag = "--use_cpu";
	const char* useCamera = "0";
	const char* useVideo = argv[3];

	if(argc == 4)
	{
		useCUDA = false;
	}
	else if((argc == 5) && (strcmp(argv[4], useCUDAFlag) == 0)){
		useCUDA = true;
	}
	else if((argc == 5) && (strcmp(argv[4], useCPUFlag) == 0)){
		useCUDA = false;
	}
	else if((argc == 5) && (strcmp(argv[4], useCUDAFlag) != 0)){
		useCUDA = false;
	}
	else
	{
		throw std::runtime_error("Too many arguments.");
	}

	if(useCUDA)
	{
		std::cout << "Inference Execution Provider: CUDA" << std::endl;
	}
	else
	{
		std::cout << "Inference Execution Provider: CPU" << std::endl;
	}

	std::string instanceName{"object-detection-inference"};
	std::string modelFilepath = argv[1];
	std::string imageFilepath = argv[2];
	std::string labelFilepath = argv[3];

	ObjectDetection *obj = new ObjectDetection(modelFilepath);

	// Ort::Env env;
	// Ort::SessionOptions session_options;
	// Ort::AllocatorWithDefaultOptions allocator;

	cv::VideoWriter video;
	cv:: Mat frame;
	cv::Mat preprocessedImage;
	cv::Mat resizedImage;

	if(!useCUDA){
		//CPU
		obj->UseCPU();
	}
	else{
		//CUDA
		obj->UseCUDA();
	}

	//Initiailizae Session
	Ort::Session session = obj->SessionInit();
	
	//Read label file
	std::vector<std::string> labels{obj->readLabels(labelFilepath)};

	if(Camera)
	{
		cv::VideoCapture cap(useVideo);
		if(!cap.isOpened()){
			return -1;
		}
		
		//Release buffer
		for(int j = 0 ; j < 10 ; j++){
			cap >> frame;
		}
		
		obj->InferenceInit(frame);

		for(int i = 0 ; i < 100 ; i++){
			
			std::cout << "Iteration : " << std::to_string(i) << std::endl;
			// //Read image file
			cap >> frame;
			
			resizedImage = obj->StaticResize(frame);
			
			
			if(i == 0){
				video = cv::VideoWriter("outcpp.avi",cv::VideoWriter::fourcc('M','J','P','G'), 30 , cv::Size(resizedImage.size().width,resizedImage.size().height));
			}

			
			//Convert Mat to Float Array
			cv::dnn::blobFromImage(resizedImage, preprocessedImage);

			
		
			start = std::chrono::system_clock::now();
			//Run Inference
			obj->RunInference(preprocessedImage);
			
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;
			std::time_t end_time = std::chrono::system_clock::to_time_t(end);
			std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
		
			obj->DrawResult(resizedImage);

			

			video.write(resizedImage);
			
		
			// cv::imshow("result", tempImg);
			// const int key = cv::waitKey(30);
			// if(key == 'q'){
			// 	break;
			// }
		}
		video.release();
		// cv::imwrite("output.jpg" , tempImg);
	}
	
	//ReleaseCUDAProviderOptions(cuda_options);
	cv::destroyAllWindows();
	return 0;
}
