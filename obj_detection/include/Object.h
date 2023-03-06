#include <iostream>
#include "opencv2/opencv.hpp"

class Object{
	public:
		cv::Rect_<float> rect_;
		int label_;
		float prob_;

};
