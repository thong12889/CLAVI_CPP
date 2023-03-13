#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/mat.hpp>
#include <vector>

class Instance{
    public:
        cv::Rect_<float> rect;
        cv::Mat *seg;
        int label;
        float prob;
};

