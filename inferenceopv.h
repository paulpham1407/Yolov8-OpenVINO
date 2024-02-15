#ifndef INFERENCEOPV_H
#define INFERENCEOPV_H

#include<string>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include <vector>
#include <random>


struct Config {
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
	std::string onnx_path;
};

struct Resize
{
	cv::Mat resized_image;
	int dw;
	int dh;
};

struct Object {
	int label{};
	float probability{};
	cv::Rect_<float> rect;
};

class InferenceOPV {
public:
	InferenceOPV(Config config);
	~InferenceOPV();
	std::vector<Object> detect(const cv::Mat& frame);

private:
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
	float rx;   // the width ratio of original image and resized image
	float ry;   // the height ratio of original image and resized image
	std::string onnx_path;
	Resize resize;
	ov::Tensor input_tensor;
	ov::InferRequest infer_request;
	ov::CompiledModel compiled_model;
	void initialmodel();
	void preprocess(cv::Mat& frame);
	std::vector<Object> postprocess(cv::Mat& frame, float* detections, ov::Shape & output_shape);
};

#endif INFERENCEOPV_H