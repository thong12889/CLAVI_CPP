#include <iostream>
#include "opencv2/opencv.hpp"

class CVQueue{
	
	public:
		CVQueue();
		CVQueue(int);
		void Enqueue(cv::Mat );
		cv::Mat Dequeue();
		int QueueSize() const;
		bool IsFull();
		bool IsEmpty();
	private:
		cv::Mat *img_q_;
		int size_;
		int front_;
		int rear_;


};
