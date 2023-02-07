#!/bin/bash
./objDetection ../utils/model/yolox_s.onnx ../utils/label/technopro_obj_labels.txt ../utils/video/object_video.mp4 --use_gpu
