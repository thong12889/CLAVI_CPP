#!/bin/bash
./classify_onnx_cpp ../../utils/cls_model/animal/model/animals_cls.onnx ../../utils/cls_model/animal/label/animals_cls_labels.txt ../../utils/cls_model/animal/test_img/cat4.jpg --use_cuda
