#include <iostream>
#include "opencv2/opencv.hpp"

class CVQueue{
	
	public:
		CVQueue();
		CVQueue(int);
		void Enqueue(cv::Mat );
		cv::Mat Dequeue();
		bool IsFull();
		bool IsEmpty();
		~CVQueue();
	private:
		cv::Mat *img_q_;
		int size_;
		int front_;
		int rear_;


};
