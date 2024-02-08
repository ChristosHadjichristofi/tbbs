#include <iostream>
#include <opencv2/opencv.hpp>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>
#include <chrono>

class ParallelImageProcessing {
private:
    const cv::Mat& inputImage;
    cv::Mat& outputImageGaussianBlur;
    cv::Mat& outputImageEdgeDetection;
    cv::Mat& outputImageGrayscale;
    int num_threads;

public:
    ParallelImageProcessing(const cv::Mat& inputImage, cv::Mat& outputImageGaussianBlur, cv::Mat& outputImageEdgeDetection, cv::Mat& outputImageGrayscale, int num_threads)
        : inputImage(inputImage), outputImageGaussianBlur(outputImageGaussianBlur), outputImageEdgeDetection(outputImageEdgeDetection), outputImageGrayscale(outputImageGrayscale), num_threads(num_threads) {}

    void applyAlgorithms() {
        oneapi::tbb::task_arena arena(num_threads); // Create task arena with specified number of threads

        // Start timer for total time
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Apply algorithms inside the task arena
        arena.execute([&] {
            // Start timer for Gaussian blur
            auto gaussian_start = std::chrono::high_resolution_clock::now();
            applyParallelGaussianBlur(1); // 1 uses a 3x3 kernel (since it includes -1, 0, +1)
            auto gaussian_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> gaussian_duration = gaussian_end - gaussian_start;
            std::cout << "Gaussian Blur Time: " << gaussian_duration.count() << " seconds\n";

            // Start timer for edge detection
            auto edge_start = std::chrono::high_resolution_clock::now();
            applyParallelEdgeDetection();
            auto edge_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> edge_duration = edge_end - edge_start;
            std::cout << "Edge Detection Time: " << edge_duration.count() << " seconds\n";

            // Start timer for grayscale conversion
            auto grayscale_start = std::chrono::high_resolution_clock::now();
            applyParallelGrayscaleConversion();
            auto grayscale_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> grayscale_duration = grayscale_end - grayscale_start;
            std::cout << "Grayscale Conversion Time: " << grayscale_duration.count() << " seconds\n";
        });

        // End timer for total time
        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_duration = total_end - total_start;
        std::cout << "Total Time: " << total_duration.count() << " seconds\n";
    }

private:
    void applyParallelGaussianBlur(int kernelSize) {
        cv::Mat tempImage(inputImage.size(), inputImage.type());
        tbb::parallel_for(tbb::blocked_range<int>(0, inputImage.rows), [&](const tbb::blocked_range<int>& r) {
            for (int y = r.begin(); y != r.end(); ++y) {
                applyGaussianBlurRowHorizontal(y, tempImage, kernelSize);
            }
        });

        tbb::parallel_for(tbb::blocked_range<int>(0, inputImage.cols), [&](const tbb::blocked_range<int>& r) {
            for (int x = r.begin(); x != r.end(); ++x) {
                applyGaussianBlurColumnVertical(x, tempImage, outputImageGaussianBlur, kernelSize);
            }
        });
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


    void applyParallelGrayscaleConversion() {
        auto grayscaleSum = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, inputImage.rows),
            0,
            [&](const tbb::blocked_range<int>& r, int localSum) {
                for (int y = r.begin(); y != r.end(); ++y) {
                    localSum += applyGrayscaleConversionRow(y);
                }
                return localSum;
            },
            [](int a, int b) {
                return a + b;
            }
        );
    }

    int applyGrayscaleConversionRow(int y) {
        int localSum = 0;
        int width = inputImage.cols;
        for (int x = 0; x < width; ++x) {
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);
            uchar grayValue = static_cast<uchar>(pixel[0] * 0.11 + pixel[1] * 0.59 + pixel[2] * 0.3);
            outputImageGrayscale.at<cv::Vec3b>(y, x) = cv::Vec3b(grayValue, grayValue, grayValue);
            localSum += grayValue;
        }
        return localSum;
    }
};

int main(int argc, char* argv[]) {
    // Check if an image path and number of threads are provided
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << " <number_of_threads>" << std::endl;
        return -1;
    }

    // Read input image
    cv::Mat inputImage = cv::imread(argv[1]);
    if (inputImage.empty()) {
        std::cout << "Could not open or find the input image." << std::endl;
        return -1;
    }

    // Parse number of threads
    const int num_threads = std::atoi(argv[2]);

    // Validate number of threads
    if (num_threads <= 0) {
        std::cerr << "Invalid number of threads\n";
        return 1;
    }

    // Create output images for each algorithm
    cv::Mat outputImageGaussianBlur = inputImage.clone();
    cv::Mat outputImageEdgeDetection = inputImage.clone();
    cv::Mat outputImageGrayscale = inputImage.clone();

    // Create ParallelImageProcessing object
    ParallelImageProcessing parallelImageProcessing(inputImage, outputImageGaussianBlur, outputImageEdgeDetection, outputImageGrayscale, num_threads);

    // Apply parallel algorithms
    parallelImageProcessing.applyAlgorithms();

    return 0;
}