#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

class ImageProcessing {
private:
    const cv::Mat& inputImage;
    cv::Mat& outputImageGaussianBlur;
    cv::Mat& outputImageEdgeDetection;
    cv::Mat& outputImageGrayscale;

public:
    ImageProcessing(const cv::Mat& inputImage, cv::Mat& outputImageGaussianBlur, cv::Mat& outputImageEdgeDetection, cv::Mat& outputImageGrayscale)
        : inputImage(inputImage), outputImageGaussianBlur(outputImageGaussianBlur), outputImageEdgeDetection(outputImageEdgeDetection), outputImageGrayscale(outputImageGrayscale) {}

    void applyAlgorithms() {
        auto startGaussian = std::chrono::high_resolution_clock::now();
        applySequentialGaussianBlur(1);
        auto endGaussian = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationGaussian = endGaussian - startGaussian;
        std::cout << "Gaussian Blur Time: " << durationGaussian.count() << " seconds\n";

        auto startEdge = std::chrono::high_resolution_clock::now();
        applySequentialEdgeDetection();
        auto endEdge = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationEdge = endEdge - startEdge;
        std::cout << "Edge Detection Time: " << durationEdge.count() << " seconds\n";

        auto startGrayscale = std::chrono::high_resolution_clock::now();
        applySequentialGrayscaleConversion();
        auto endGrayscale = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationGrayscale = endGrayscale - startGrayscale;
        std::cout << "Grayscale Conversion Time: " << durationGrayscale.count() << " seconds\n";
    }

private:
    void applySequentialGaussianBlur(int kernelSize) {
        cv::Mat tempImage(inputImage.size(), inputImage.type());
        for (int y = 0; y < inputImage.rows; ++y) {
            applyGaussianBlurRowHorizontal(y, tempImage, kernelSize);
        }
        for (int x = 0; x < inputImage.cols; ++x) {
            applyGaussianBlurColumnVertical(x, tempImage, outputImageGaussianBlur, kernelSize);
        }
    }

    void applyGaussianBlurRowHorizontal(int y, cv::Mat& tempImage, int kernelSize) const {
        int width = inputImage.cols;
        for (int x = 0; x < width; ++x) {
            tempImage.at<cv::Vec3b>(y, x) = applyGaussianKernelHorizontal(inputImage, x, y, kernelSize);
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

    void applyGaussianBlurColumnVertical(int x, const cv::Mat& tempImage, cv::Mat& outputImage, int kernelSize) const {
        int height = inputImage.rows;
        for (int y = 0; y < height; ++y) {
            outputImage.at<cv::Vec3b>(y, x) = applyGaussianKernelVertical(tempImage, x, y, kernelSize);
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

    void applySequentialEdgeDetection() {
        cv::Mat grayscaleImage;
        cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);
        cv::Mat edges = cv::Mat::zeros(grayscaleImage.size(), CV_8UC1);
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        for (int y = 1; y < grayscaleImage.rows - 1; ++y) {
            for (int x = 1; x < grayscaleImage.cols - 1; ++x) {
                int sumX = 0, sumY = 0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        sumX += grayscaleImage.at<uchar>(y + i, x + j) * Gx[i + 1][j + 1];
                        sumY += grayscaleImage.at<uchar>(y + i, x + j) * Gy[i + 1][j + 1];
                    }
                }
                int sum = abs(sumX) + abs(sumY);
                edges.at<uchar>(y, x) = sum > 255 ? 255 : sum;
            }
        }
        edges.copyTo(outputImageEdgeDetection);
    }

    void applySequentialGrayscaleConversion() {
        for (int y = 0; y < inputImage.rows; ++y) {
            for (int x = 0; x < inputImage.cols; ++x) {
                cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);
                uchar grayValue = static_cast<uchar>(pixel[0] * 0.11 + pixel[1] * 0.59 + pixel[2] * 0.3);
                outputImageGrayscale.at<cv::Vec3b>(y, x) = cv::Vec3b(grayValue, grayValue, grayValue);
            }
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

    // Create ImageProcessing object and apply algorithms
    ImageProcessing imageProcessing(inputImage, outputImageGaussianBlur, outputImageEdgeDetection, outputImageGrayscale);
    imageProcessing.applyAlgorithms();

    // Display the original and processed images
    cv::imshow("Original Image", inputImage);
    cv::imshow("Gaussian Blur", outputImageGaussianBlur);
    cv::imshow("Edge Detection", outputImageEdgeDetection);
    cv::imshow("Grayscale Conversion", outputImageGrayscale);

    // Wait for a key press before exiting
    cv::waitKey(0);

    return 0;
}