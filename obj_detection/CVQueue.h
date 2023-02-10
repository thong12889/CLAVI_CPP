#ifndef Queue_h
#define Queue_h

#include <iostream>
#include "opencv2/opencv.hpp"
// #include "Node.h"

class CVQueue{
	
	public:
		CVQueue();
		CVQueue(int);
		void Enqueue(cv::Mat& );
		cv::Mat Dequeue();
		bool IsFull();
		bool IsEmpty();
		void Display();
		int GetRear() const;
		int GetFront() const;
		cv::Mat *GetQAddr() const;
	private:
		cv::Mat *img_q_;
		cv::Mat temp_;
		int size_;
		int front_;
		int rear_;


};

#endif
