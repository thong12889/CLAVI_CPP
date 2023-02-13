#!/bin/bash
./objDetection ../utils/model/technopro_obj.onnx ../utils/label/technopro_obj_labels.txt ../utils/video/object_video.mp4 --use_cuda
# ./objDetection ../utils/model/yolox_s.onnx ../utils/label/technopro_obj_labels.txt ../utils/video/object_video.mp4 --use_cuda