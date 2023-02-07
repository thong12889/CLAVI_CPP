#include <iostream>
#include "opencv2/opencv.hpp"
#include "CVQueue.h"

using namespace std;

CVQueue::CVQueue(){
}

CVQueue::CVQueue(int size){
	this->img_q_ = new cv::Mat[size+1];
	this->size_ = size+1;
	this->front_ = 1;
	this->rear_ = 1;
}

void CVQueue::Enqueue(cv::Mat q){
	if(this->IsFull()){
		std::cout << "Queue is full" << std::endl;
	}
	else{
		img_q_[this->rear_] = q.clone();
		this->rear_ = (this->rear_ % this->size_)+1;
	}
	
}

cv::Mat CVQueue::Dequeue(){
	cv::Mat temp;
	if(IsEmpty()){
		std::cout << "Queue is empty" << std::endl;
	}
	else{
		temp = this->img_q_[this->front_].clone();
		this->img_q_[this->rear_].release();
		this->front_ = (this->front_ % this->size_) +1;
	}
	return temp;
}

bool CVQueue::IsEmpty(){
	return this->rear_ == this->front_;
}

bool CVQueue::IsFull(){
	return (this->rear_ % this->size_) +1 == this->front_;
}

CVQueue::~CVQueue(){
}

