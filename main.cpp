#include "inferenceopv.h"

int main(int argc, char* argv[]) {
    
    try{
        const std::string input_model_path = "yolov8s.xml";
        const std::string input_image_path = "original.jpg";

        clock_t start, end;
        cv::Mat img = cv::imread(input_image_path);

        InferenceOPV model(input_model_path);
        start = clock();
        
        std::vector<Object> detections = model.detect(img);

        for (const auto& detection : detections) {
            std::cout << "  Object: " << std::endl;
            std::cout << "  Label: " << detection.class_id << std::endl;
            std::cout << "  Probability: " << detection.confidence << std::endl;
            std::cout << "  Rect: " << detection.box << std::endl;
        }
    }catch (const std::exception& ex){
        std::cerr << ex.what()<<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
