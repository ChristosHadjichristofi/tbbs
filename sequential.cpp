#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

class SequentialImageProcessing {
private:
    const cv::Mat& inputImage;
    cv::Mat& outputImageGaussianBlur;
    cv::Mat& outputImageEdgeDetection;
    cv::Mat& outputImageGrayscale;

public:
    SequentialImageProcessing(const cv::Mat& inputImage, cv::Mat& outputImageGaussianBlur, cv::Mat& outputImageEdgeDetection, cv::Mat& outputImageGrayscale)
        : inputImage(inputImage), outputImageGaussianBlur(outputImageGaussianBlur), outputImageEdgeDetection(outputImageEdgeDetection), outputImageGrayscale(outputImageGrayscale) {}

    void applyAlgorithms() const {
        // Example: Apply sequential Gaussian blur
        auto startGaussian = std::chrono::high_resolution_clock::now();
        applySequentialGaussianBlur();
        auto endGaussian = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationGaussian = endGaussian - startGaussian;
        std::cout << "Sequential Gaussian Blur Time: " << durationGaussian.count() << " seconds\n";

        // Example: Apply sequential edge detection
        auto startEdge = std::chrono::high_resolution_clock::now();
        applySequentialEdgeDetection();
        auto endEdge = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationEdge = endEdge - startEdge;
        std::cout << "Sequential Edge Detection Time: " << durationEdge.count() << " seconds\n";

        // Example: Apply sequential grayscale conversion
        auto startGrayscale = std::chrono::high_resolution_clock::now();
        applySequentialGrayscaleConversion();
        auto endGrayscale = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationGrayscale = endGrayscale - startGrayscale;
        std::cout << "Sequential Grayscale Conversion Time: " << durationGrayscale.count() << " seconds\n";

        // Add more sequential algorithms here
        // applySequentialAlgorithm4();
    }

private:
    void applySequentialGaussianBlur() const {
        cv::Mat tempImage(inputImage.size(), inputImage.type());

        // Apply horizontal pass
        for (int y = 0; y < inputImage.rows; ++y) {
            applyGaussianBlurRowHorizontal(y, tempImage);
        }

        // Apply vertical pass
        for (int x = 0; x < inputImage.cols; ++x) {
            applyGaussianBlurColumnVertical(x, tempImage);
        }

        tempImage.copyTo(outputImageGaussianBlur);
    }

    void applyGaussianBlurRowHorizontal(int y, cv::Mat& tempImage) const {
        int width = inputImage.cols;

        for (int x = 0; x < width; ++x) {
            tempImage.at<cv::Vec3b>(y, x) = applyGaussianKernelHorizontal(inputImage, x, y, 3);
        }
    }

    cv::Vec3b applyGaussianKernelHorizontal(const cv::Mat& image, int x, int y, int kernelSize) const {
        cv::Vec3f sum(0, 0, 0);
        float totalWeight = 0;

        for (int j = -kernelSize; j <= kernelSize; ++j) {
            int currentX = std::min(std::max(x + j, 0), image.cols - 1);
            float weight = std::exp(-(j * j) / (2.0 * kernelSize * kernelSize));
            sum += static_cast<cv::Vec3f>(image.at<cv::Vec3b>(y, currentX)) * weight;
            totalWeight += weight;
        }

        return cv::Vec3b(sum[0] / totalWeight, sum[1] / totalWeight, sum[2] / totalWeight);
    }

    void applyGaussianBlurColumnVertical(int x, cv::Mat& tempImage) const {
        int height = inputImage.rows;

        for (int y = 0; y < height; ++y) {
            outputImageGaussianBlur.at<cv::Vec3b>(y, x) = applyGaussianKernelVertical(tempImage, x, y, 3);
        }
    }

    cv::Vec3b applyGaussianKernelVertical(const cv::Mat& image, int x, int y, int kernelSize) const {
        cv::Vec3f sum(0, 0, 0);
        float totalWeight = 0;

        for (int i = -kernelSize; i <= kernelSize; ++i) {
            int currentY = std::min(std::max(y + i, 0), image.rows - 1);
            float weight = std::exp(-(i * i) / (2.0 * kernelSize * kernelSize));
            sum += static_cast<cv::Vec3f>(image.at<cv::Vec3b>(currentY, x)) * weight;
            totalWeight += weight;
        }

        return cv::Vec3b(sum[0] / totalWeight, sum[1] / totalWeight, sum[2] / totalWeight);
    }

    void applySequentialEdgeDetection() const {
        cv::Mat grayscaleImage;
        cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

        cv::Mat edges(inputImage.size(), CV_8UC1, cv::Scalar(0));

        for (int y = 0; y < inputImage.rows; ++y) {
            applyEdgeDetectionRow(y, edges);
        }

        edges.copyTo(outputImageEdgeDetection);
    }

    void applyEdgeDetectionRow(int y, cv::Mat& edges) const {
        int width = inputImage.cols;

        for (int x = 0; x < width; ++x) {
            // Check if the pixel is an edge
            if (edges.at<uchar>(y, x) > 0) {
                // Gray out edge pixels
                outputImageEdgeDetection.at<cv::Vec3b>(y, x) = cv::Vec3b(128, 128, 128); // Gray color for edges
            } else {
                // Make other pixels black
                outputImageEdgeDetection.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    void applySequentialGrayscaleConversion() const {
        for (int y = 0; y < inputImage.rows; ++y) {
            applyGrayscaleConversionRow(y);
        }
    }

    void applyGrayscaleConversionRow(int y) const {
        int width = inputImage.cols;

        for (int x = 0; x < width; ++x) {
            // Perform grayscale conversion
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);
            uchar grayValue = static_cast<uchar>((pixel[0] + pixel[1] + pixel[2]) / 3);

            outputImageGrayscale.at<cv::Vec3b>(y, x) = cv::Vec3b(grayValue, grayValue, grayValue);
        }
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

    // Create SequentialImageProcessing object
    SequentialImageProcessing sequentialImageProcessing(inputImage, outputImageGaussianBlur, outputImageEdgeDetection, outputImageGrayscale);

    // Apply sequential algorithms
    sequentialImageProcessing.applyAlgorithms();

    // Display the original and processed images for each algorithm
    cv::imshow("Original Image", inputImage);
    cv::imshow("Sequential Gaussian Blur", outputImageGaussianBlur);
    cv::imshow("Sequential Edge Detection", outputImageEdgeDetection);
    cv::imshow("Sequential Grayscale Conversion", outputImageGrayscale);
    cv::waitKey(0);

    return 0;
}
