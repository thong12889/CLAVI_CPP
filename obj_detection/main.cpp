#include <iostream>
#include "ObjectDetection.h"
#include <vector>

#include <iterator>
#include <memory>
#include <string>

#include "CVQueue.h"

template <typename T> T vectorProduct(std::vector<T>& v){
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


int main(int argc, char* argv[]){
	ObjectDetection *obj = new ObjectDetection();

	if(argc < 4)
	{
		std::cerr << "Usage CPU: [apps] [path/to/model] [path/to/image] [path/to/labal]" << std::endl; 
		std::cerr <<  "Usage GPU: [apps] [path/to/model] [path/to/image] [path/to/label] --use_cuda" << std::endl;
		return EXIT_FAILURE;
	}

	bool Camera = true;
	bool Onetime = false;
	bool useCUDA = false;
	const char* useCUDAFlag = "--use_gpu";
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

	Ort::Env env;
	Ort::SessionOptions session_options;
	Ort::AllocatorWithDefaultOptions allocator;
	if(!useCUDA){
		session_options.SetIntraOpNumThreads(1);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	}
	else{
		//CUDA
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = 0;
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch();
		cuda_options.cuda_mem_limit = static_cast<int>(SIZE_MAX * 1024 * 1024);
		cuda_options.arena_extend_strategy = 1;
		cuda_options.do_copy_in_default_stream = 1;
		session_options.AppendExecutionProvider_CUDA(cuda_options);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	}

	Ort::Session session(env, modelFilepath.c_str(), session_options);

	size_t numInputNodes = session.GetInputCount();
	size_t numOutputNodes = session.GetOutputCount();

	const char* inputName = session.GetInputName(0, allocator);

	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);	
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

	std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

	const char* outputName0 = session.GetOutputName(0, allocator);
	const char* outputName1 = session.GetOutputName(1, allocator);

	std::vector<const char*> inputNames{inputName};
	std::vector<const char*> outputNames{outputName0, outputName1};
	std::vector<Ort::Value> inputTensors;

	if(Camera)
	{
		cv::VideoCapture cap(useVideo);
		if(!cap.isOpened()){
			return -1;
		}
		
		cv:: Mat frame;
		cv::Mat preprocessedImage;

		//Read label file
		std::vector<std::string> labels{obj->readLabels(labelFilepath)};

		int img_w;
		int img_h;
		float scale;
		cv::Mat resizedImage, tempImg;
		size_t inputTensorSize = vectorProduct(inputDims);
		std::vector<float> inputTensorValues(inputTensorSize);
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		std::vector<Ort::Value> outputTensorValues;

		const float* pred;
		std::vector<long> pred_dim;
		const int64_t* label;
		std::vector<long> label_dim;
		

		//OpenCV Miscellaneous Function
		//Release buffer
		for(int j = 0 ; j < 10 ; j++){
			cap >> frame;
		}
		tempImg = obj->StaticResize(frame);
		cv::VideoWriter video = cv::VideoWriter("outcpp.avi",cv::VideoWriter::fourcc('M','J','P','G'), 30 , cv::Size(tempImg.size().width,tempImg.size().height));
		
		for(int i = 0 ; i < 1000 ; i++){
			// //Read image file
			cap >> frame;
			

			img_w = frame.cols;
			img_h = frame.rows;
			scale = std::min(obj->GetInputW() / (frame.cols * 1.0), obj->GetInputH() / (frame.rows * 1.0));
			std::vector<Object> objects;


			resizedImage = obj->StaticResize(frame);
			tempImg = resizedImage.clone();
			cv::dnn::blobFromImage(resizedImage, preprocessedImage);
		
			
			inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
		
			
			inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
		
			outputTensorValues = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), numInputNodes, outputNames.data(), numOutputNodes);
		
			
			pred = outputTensorValues[0].GetTensorMutableData<float>();
			pred_dim = outputTensorValues[0].GetTensorTypeAndShapeInfo().GetShape();
			label = outputTensorValues[1].GetTensorMutableData<int64_t>();
			label_dim = outputTensorValues[1].GetTensorTypeAndShapeInfo().GetShape();
			
			// //Get results
			obj->decode_outputs(pred, pred_dim, label, objects, scale, img_w, img_h);
			
			// obj->nms_sorted_bboxes(objects);
			
			obj->DrawResult(objects, tempImg, labels);

			
			
			video.write(tempImg);
			
		
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
