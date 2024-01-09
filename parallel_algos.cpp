#include <iostream>
#include <opencv2/opencv.hpp>
#include <tbb/tbb.h>
#include <chrono>

class ParallelImageProcessing {
private:
    const cv::Mat& inputImage;
    cv::Mat& outputImageGaussianBlur;
    cv::Mat& outputImageEdgeDetection;
    cv::Mat& outputImageGrayscale;

public:
    ParallelImageProcessing(const cv::Mat& inputImage, cv::Mat& outputImageGaussianBlur, cv::Mat& outputImageEdgeDetection, cv::Mat& outputImageGrayscale)
        : inputImage(inputImage), outputImageGaussianBlur(outputImageGaussianBlur), outputImageEdgeDetection(outputImageEdgeDetection), outputImageGrayscale(outputImageGrayscale) {}

    void applyAlgorithms() const {
        // Example: Apply parallel Gaussian blur
        auto startGaussian = std::chrono::high_resolution_clock::now();
        applyParallelGaussianBlur();
        auto endGaussian = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationGaussian = endGaussian - startGaussian;
        std::cout << "Gaussian Blur Time: " << durationGaussian.count() << " seconds\n";

        // Example: Apply parallel edge detection
        auto startEdge = std::chrono::high_resolution_clock::now();
        applyParallelEdgeDetection();
        auto endEdge = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationEdge = endEdge - startEdge;
        std::cout << "Edge Detection Time: " << durationEdge.count() << " seconds\n";

        // Example: Apply parallel grayscale conversion
        auto startGrayscale = std::chrono::high_resolution_clock::now();
        applyParallelGrayscaleConversion();
        auto endGrayscale = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationGrayscale = endGrayscale - startGrayscale;
        std::cout << "Grayscale Conversion Time: " << durationGrayscale.count() << " seconds\n";

        // Add more parallel algorithms here
        // applyParallelAlgorithm4();
    }

private:
    void applyParallelGaussianBlur() const {
        cv::Mat tempImage(inputImage.size(), inputImage.type());

        // Apply horizontal pass
        tbb::parallel_for(tbb::blocked_range<int>(0, inputImage.rows), [&](const tbb::blocked_range<int>& r) {
            for (int y = r.begin(); y != r.end(); ++y) {
                applyGaussianBlurRowHorizontal(y, tempImage);
            }
        });

        // Apply vertical pass directly to outputImageGaussianBlur
        tbb::parallel_for(tbb::blocked_range<int>(0, inputImage.cols), [&](const tbb::blocked_range<int>& r) {
            for (int x = r.begin(); x != r.end(); ++x) {
                applyGaussianBlurColumnVertical(x, tempImage, outputImageGaussianBlur);
            }
        });
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

    void applyGaussianBlurColumnVertical(int x, const cv::Mat& tempImage, cv::Mat& outputImage) const {
        int height = inputImage.rows;

        for (int y = 0; y < height; ++y) {
            outputImage.at<cv::Vec3b>(y, x) = applyGaussianKernelVertical(tempImage, x, y, 3);
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

    void applyParallelEdgeDetection() const {
    cv::Mat grayscaleImage;
    cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    cv::Mat edges = cv::Mat::zeros(grayscaleImage.size(), CV_8UC1);

    // Define Sobel kernels for simplicity
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    tbb::parallel_for(tbb::blocked_range<int>(1, grayscaleImage.rows-1), [&](const tbb::blocked_range<int>& r) {
        for (int y = r.begin(); y != r.end(); ++y) {
            for (int x = 1; x < grayscaleImage.cols-1; x++) {
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
    });

    edges.copyTo(outputImageEdgeDetection);
}


    void applyParallelGrayscaleConversion() const {
        tbb::parallel_for(tbb::blocked_range<int>(0, inputImage.rows), [&](const tbb::blocked_range<int>& r) {
            for (int y = r.begin(); y != r.end(); ++y) {
                applyGrayscaleConversionRow(y);
            }
        });
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

    // Create ParallelImageProcessing object
    ParallelImageProcessing parallelImageProcessing(inputImage, outputImageGaussianBlur, outputImageEdgeDetection, outputImageGrayscale);

    // Apply parallel algorithms
    parallelImageProcessing.applyAlgorithms();

    // Display the original and processed images for each algorithm
    cv::imshow("Original Image", inputImage);
    cv::imshow("Gaussian Blur", outputImageGaussianBlur);
    cv::imshow("Edge Detection", outputImageEdgeDetection);
    cv::imshow("Grayscale Conversion", outputImageGrayscale);
    cv::waitKey(0);

    return 0;
}
