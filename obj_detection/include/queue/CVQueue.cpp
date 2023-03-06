#include <iostream>
#include "CVQueue.h"
#include "opencv2/opencv.hpp"

CVQueue::CVQueue(){}
CVQueue::CVQueue(int size){
	this->img_q_ = new cv::Mat[size];
	this->size_ = size;
	this->front_ = -1;
	this->rear_ = -1;
}
void CVQueue::Enqueue(cv::Mat& frame){
	// if(this->IsFull()){
	// 	std::cout << "This queue is full" << std::endl;
	// }
	// else{
	// 	this->rear_ = (this->rear_+1)%this->size_;
	// 	this->q_[this->rear_] = q;
	// }
	if(this->IsFull()){
		this->Dequeue();
	}
	this->rear_ = (this->rear_ +1) % this->size_;
	this->img_q_[this->rear_] = frame;
	if(this->front_ == -1){
		this->front_ ++;
	}
	// if(this->IsFull()){
	// 	std::cout << "Queue is Full" << std::endl;
	// }
	// else{
	// 	this->rear_ = (this->rear_ +1) % this->size_;
	// 	this->img_q_[this->rear_] = frame;
	// 	if(this->front_ == -1){
	// 		this->front_ ++;
	// 	}

	// }
}

bool CVQueue::IsFull(){
	// return (this->rear_%this->size_)+1 == this->front_;
	// return this->rear_ == this->front_;
	return (this->rear_ +1) % this->size_ == this->front_;
}
bool CVQueue::IsEmpty(){
	return this->front_ == -1;
	
}

cv::Mat CVQueue::Dequeue(){
	// if(this->IsEmpty()){
	// 	std::cout << "This Queue is empty" << std::endl;
	// }
	// else{
	// 	this->front_ = (this->front_+1)%this->size_;
	// 	x = this->q_[this->front_];
	// 	// x = this->q_[this->front_];
	// 	this->q_[this->front_] = 0;
	// 	// this->front_ = (this->front_ % this->size_) +1;
		
	// }
	if(this->IsEmpty()){
		std::cout << "Queue is Empty" << std::endl;
	}
	else{
		if(this->front_ == this->rear_){
			
			//Copy frame to temporary frame
			this->temp_ = this->img_q_[this->front_].clone();
			//Clear frame in queue as empty frame
			this->img_q_[this->front_].release();
			this->front_ = (this->front_ + 1) % this->size_;
			//Update When queue has left 1 item, Reset to the start as Rear = -1 and Front = -1 
			this->front_ = this->rear_ = -1;
			
		}
		else{
			
			this->temp_ = this->img_q_[this->front_].clone();
			this->img_q_[this->front_].release();
			this->front_ = (this->front_ + 1) % this->size_;
		}
		
		
	}
	return this->temp_;

}



void CVQueue::Display(){
	for(int i =0; i < size_; i++){
		if(!this->img_q_[i].empty()){
			std::cout << "1 " ;
		}
		else{
			std::cout << "0 " ;
		}
		
	}
	std::cout << "" << std::endl;
	// for(int i = 0 ; i < this->size_ ; i++){
	// 	std::cout << this->q_[i] << " " ;
	// }
}

int CVQueue::GetRear() const{
	return this->rear_;
}

int CVQueue::GetFront() const{
	return this->front_;
}

cv::Mat * CVQueue::GetQAddr() const{
	return this->img_q_;
}