#include <iostream>
#include "opencv2/opencv.hpp"
#include "CVQueue.h"
#include <thread>

#include <iterator>
#include <memory>
#include <string>

#include "ObjectDetection.h"

#include <onnxruntime_cxx_api.h>

CVQueue *q = new CVQueue(10);
cv::Mat img_resize;

int const height = 1080;
int const width = 1920;
int h_resize , w_resize;
cv::VideoWriter video;
bool video_save = false;
ObjectDetection *obj;
cv::VideoCapture cap(0); 

void ThreadQueue(){
	cv::Mat frame;
	while(cap.isOpened()){
		cap >> frame;
		if(frame.empty()){
			std::cout << "Application closed due to empty frame" << std::endl;
		}
		else{
			q->Enqueue(frame);
			std::cout << "Enqueue" << std::endl;
		}
	}
	

}

cv::Mat *ImgResize(cv::Mat img){
	h_resize = height/img.size().height;
	w_resize = width/img.size().width;
	cv::resize(img, img_resize , cv::Size() , h_resize, w_resize);
	return &img_resize;
}

void ThreadDisplay(){
	// cv::Mat *img;
	cv::Mat temp, show_img, resizedImage, preprocessedImage;
	Ort::Session session = obj->SessionInit();

	while(1){
		
		if(!q->IsEmpty()){
			q->Dequeue().copyTo(temp);
			cv::resize(temp, resizedImage, cv::Size(640, 640));
			obj->InferenceInit(resizedImage);
			std::cout << "Init Inference" << std::endl;
			break;
		}
	}
	
	
	while(1){
		if(!q->IsEmpty()){
			q->Dequeue().copyTo(temp);
			if(!temp.empty()){
				cv::resize(temp, resizedImage, cv::Size(640, 640));
				cv::dnn::blobFromImage(resizedImage, preprocessedImage);
				obj->RunInference(preprocessedImage);
				obj->DrawResult(resizedImage);

				std::cout << "Get Frame" << std::endl;
				cv::imshow("RTP" , resizedImage); 
				if(cv::waitKey(1) == 'q'){
					break;
				}
			}
			else{
				std::cout << "Empty Frame" << std::endl;
			}
		}
	}
	cap.release();
}

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
	std::string labelFilepath = argv[2];
	std::string imageFilepath = argv[3];

	obj = new ObjectDetection(modelFilepath, labelFilepath);

	if(!useCUDA){
		//CPU
		obj->UseCPU();
	}
	else{
		//CUDA
		obj->UseCUDA();
		std::cout << "Use CUDA" << std::endl;
	}

	//Thread Affinity
	//Create CPU Set for ThreadQueue and  ThreadDisplay
	cpu_set_t cpu_set_1;
	cpu_set_t cpu_set_2;

	//Zero Them out
	CPU_ZERO(&cpu_set_1);
	CPU_ZERO(&cpu_set_2);

	//Set CPU Cores we want to pin thread too
	CPU_SET(0,&cpu_set_1);
	CPU_SET(1,&cpu_set_2);

	std::thread queue_thread(ThreadQueue);
	//Pin ThreadQueue to CPU Core 0
	assert(pthread_setaffinity_np(queue_thread.native_handle(), sizeof(cpu_set_t), &cpu_set_1) == 0);
	std::thread display_thread(ThreadDisplay);
	//Pin ThreadDisplay to CPU Core 1
	assert(pthread_setaffinity_np(display_thread.native_handle(), sizeof(cpu_set_t), &cpu_set_2) == 0);
	queue_thread.join();
	display_thread.join();
}

