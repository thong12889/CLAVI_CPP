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

void ThreadQueue(){
	char env[] = "OPENCV_FFMPEG_CAPTURE_OPTIONS=protocol_whitelist;file,rtp,udp";
	int rc = putenv(env);
    if (rc != 0){
		std::cout << "Could not set environment variables\n" << std::endl;
    }
	char input[] = "../utils/sdp/ffmpegextension.sdp";
	cv::VideoCapture cap(input , cv::CAP_FFMPEG);
	cv::Mat frame;
      if(!cap.isOpened()){
             std::cout << "Error opening video stream or file" << std::endl;
      }
	cap >> frame;

	for(;;){
		cap >> frame;
		q->Enqueue(frame);
	}
}

void ThreadDisplay(){
	// cv::Mat *img;
	cv::Mat temp, show_img, resizedImage, preprocessedImage;
	Ort::Session session = obj->SessionInit();

	while(1){
		
		if(!q->IsEmpty()){
			temp = q->Dequeue();
			cv::resize(temp, resizedImage, cv::Size(640, 640));
			obj->InferenceInit(resizedImage);
			std::cout << "Init Inference" << std::endl;
			
			break;
		}
	}
	
	
	while(1){
		if(!q->IsEmpty()){
			// img = ImgResize(q->Dequeue());
			temp = q->Dequeue();
			
			

			if(!video_save){
				video_save = true;
				video = cv::VideoWriter("out.avi",cv::VideoWriter::fourcc('M','J','P','G'),30, cv::Size(1280,720),true);
			}
			if(!temp.empty()){
				cv::resize(temp, resizedImage, cv::Size(640, 640));
				cv::dnn::blobFromImage(resizedImage, preprocessedImage);
				obj->RunInference(preprocessedImage);
				obj->DrawResult(resizedImage);

				std::cout << "Get Frame" << std::endl;
				if(video_save){
					cv::resize(resizedImage, show_img , cv::Size(1280,720));
					video.write(show_img);
				}
				// cv::imshow("RTP" , resizedImage); 
				// if(cv::waitKey(1) == 'q'){
				// 	break;
				// }
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

	obj = new ObjectDetection(modelFilepath,labelFilepath);

	if(!useCUDA){
		//CPU
		obj->UseCPU();
	}
	else{
		//CUDA
		obj->UseCUDA();
		std::cout << "Use CUDA" << std::endl;
	}

	


	std::thread queue_thread(ThreadQueue);
	std::thread display_thread(ThreadDisplay);
	queue_thread.join();
	display_thread.join();
}

