#pragma once
#include <string>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include <vector>
#include <random>

struct Resize
{
	cv::Mat resized_image;
	int dw;
	int dh;
};

struct Object {
	int class_id{ 0 };
	std::string class_name{};
	float confidence{ 0.0 };
	cv::Rect_ <float> box{};
	cv::Mat mask{};
};

class InferenceOPV {
public:
	InferenceOPV(std::string model_path, const int height = 640, const int width = 640, const float m_proThreshold = 0.50f, const float nmsThreshold = 0.65f);
	std::vector <Object> detect(const cv::Mat& inputImgBGR);
private:
	float rx;
	float ry;
	int m_height;
	int m_width;
	float m_proThreshold;
	float m_nmsThreshold;
	Resize resize;
	ov::Tensor input_tensor;
	ov::InferRequest infer_request;
	ov::CompiledModel compiled_model;

	void preprocess(const cv::Mat& frame);
	std::vector <Object> postprocess(const cv::Mat& frame, float* detections, ov::Shape& output_shape);
};