#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

class OpenCVImageProcessing {
private:
    const cv::Mat& inputImage;
    cv::Mat& outputImageGaussianBlur;
    cv::Mat& outputImageEdgeDetection;
    cv::Mat& outputImageGrayscale;

public:
    OpenCVImageProcessing(const cv::Mat& inputImage, cv::Mat& outputImageGaussianBlur, cv::Mat& outputImageEdgeDetection, cv::Mat& outputImageGrayscale)
        : inputImage(inputImage), outputImageGaussianBlur(outputImageGaussianBlur), outputImageEdgeDetection(outputImageEdgeDetection), outputImageGrayscale(outputImageGrayscale) {}

    void applyAlgorithms() const {
        // Example: Apply OpenCV Gaussian blur
        auto startGaussian = std::chrono::high_resolution_clock::now();
        cv::GaussianBlur(inputImage, outputImageGaussianBlur, cv::Size(3, 3), 0, 0);
        auto endGaussian = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationGaussian = endGaussian - startGaussian;
        std::cout << "OpenCV Gaussian Blur Time: " << durationGaussian.count() << " seconds\n";

        // Example: Apply OpenCV edge detection (Canny)
        auto startEdge = std::chrono::high_resolution_clock::now();
        cv::Mat grayscaleImage;
        cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);
        cv::Canny(grayscaleImage, outputImageEdgeDetection, 50, 150);
        auto endEdge = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationEdge = endEdge - startEdge;
        std::cout << "OpenCV Edge Detection Time: " << durationEdge.count() << " seconds\n";

        // Example: Apply OpenCV grayscale conversion
        auto startGrayscale = std::chrono::high_resolution_clock::now();
        cv::cvtColor(inputImage, outputImageGrayscale, cv::COLOR_BGR2GRAY);
        auto endGrayscale = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationGrayscale = endGrayscale - startGrayscale;
        std::cout << "OpenCV Grayscale Conversion Time: " << durationGrayscale.count() << " seconds\n";
    }
};

int main(int argc, char* argv[]) {
    // Check if an image path is provided
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Read input image
    cv::Mat inputImage = cv::imread(argv[1]);
    if (inputImage.empty()) {
        std::cout << "Could not open or find the input image." << std::endl;
        return -1;
    }

    // Create output images for each algorithm
    cv::Mat outputImageGaussianBlur = inputImage.clone();
    cv::Mat outputImageEdgeDetection = inputImage.clone();
    cv::Mat outputImageGrayscale = inputImage.clone();

    // Create OpenCVImageProcessing object
    OpenCVImageProcessing opencvImageProcessing(inputImage, outputImageGaussianBlur, outputImageEdgeDetection, outputImageGrayscale);

    // Apply OpenCV algorithms
    opencvImageProcessing.applyAlgorithms();

    // Display the original and processed images for each algorithm
    // cv::imshow("Original Image", inputImage);
    // cv::imshow("OpenCV Gaussian Blur", outputImageGaussianBlur);
    // cv::imshow("OpenCV Edge Detection", outputImageEdgeDetection);
    // cv::imshow("OpenCV Grayscale Conversion", outputImageGrayscale);
    // cv::waitKey(0);

    return 0;
}
