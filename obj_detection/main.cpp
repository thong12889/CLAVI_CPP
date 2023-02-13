#include <iostream>
#include "opencv2/opencv.hpp"
#include "CVQueue.h"
#include <thread>

#include <iterator>
#include <memory>
#include <string>

#include "ObjectDetection.h"

#include <onnxruntime_cxx_api.h>

CVQueue *q = new CVQueue(1);
cv::Mat img_resize;

int const height = 1080;
int const width = 1920;
int h_resize , w_resize;
cv::VideoWriter video;
bool video_save = false;
ObjectDetection *obj;

void ThreadQueue(){
	cv::VideoCapture cap(0); 
	cv::Mat frame;
	while(1){
		cap >> frame;
		if(frame.empty()){
			std::cout << "Application closed due to empty frame" << std::endl;
		}
		else{
			q->Enqueue(frame);
			std::cout << "Enqueue" << std::endl;
		}
	}
	
	if(!cap.isOpened()){
			std::cout << "Error opening video stream or file" << std::endl;
	}

	while(cap.isOpened()){
		cap >> frame;
		if(!frame.empty()){
			q->Enqueue(frame);
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
	cv::VideoCapture cap(0); 
	cv::Mat frame;
	cv::Mat temp, show_img, resizedImage, preprocessedImage;
	Ort::Session session = obj->SessionInit();

	while(1){
		cap >> temp;
		if(!temp.empty()){
			// temp = q->Dequeue();
			
			cv::resize(temp, resizedImage, cv::Size(640, 640));
			obj->InferenceInit(resizedImage);
			std::cout << "Init Inference" << std::endl;
			
			break;
		}
	}
	
	
	while(1){
		cap >> temp;
		if(!temp.empty()){
			// img = ImgResize(q->Dequeue());
			// temp = q->Dequeue();
			
			// if(video_save){
			// 	video = cv::VideoWriter("out.avi",cv::VideoWriter::fourcc('M','J','P','G'),30, cv::Size(1280,720),true);
			// }
			if(!temp.empty()){
				cv::resize(temp, resizedImage, cv::Size(640, 640));
				cv::dnn::blobFromImage(resizedImage, preprocessedImage);
				obj->RunInference(preprocessedImage);
				obj->DrawResult(resizedImage);

				std::cout << "Get Frame" << std::endl;
				// if(video.isOpened()){
				// 	cv::resize(resizedImage, show_img , cv::Size(1280,720));
				// 	video.write(show_img);
				// }
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

	

	// std::thread display_thread(ThreadDisplay);
	// std::thread queue_thread(ThreadQueue);
	// display_thread.join();
	ThreadDisplay();
	// queue_thread.join();
	
}

