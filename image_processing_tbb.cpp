#include <iostream>
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <atomic>

namespace fs = std::filesystem;
using namespace std::chrono;

class ImageProcessingTask {
private:
    std::string outputFolder;
    mutable std::atomic<int> progress; // Mutable to modify in const member function

public:
    ImageProcessingTask(const std::string& outputFolder) : outputFolder(outputFolder), progress(0) {}

    void operator()(const fs::directory_entry& entry) const {
        std::string inputImagePath = entry.path().string();
        std::string outputImageBaseName = entry.path().stem().string();

        // Read input image
        cv::Mat inputImage = cv::imread(inputImagePath);
        if (inputImage.empty()) {
            std::cout << "Could not open or find the input image: " << inputImagePath << std::endl;
            return;
        }

        // Process the image with all algorithms
        processImageWithAllAlgorithms(inputImage, outputImageBaseName);

        // Increment progress
        ++progress;
    }

    void processImageWithAllAlgorithms(cv::Mat& inputImage, const std::string& outputBaseName) const {
        // Apply image processing algorithms here
        cv::Mat outputBlur, outputGrayscale, outputEdges, outputSegmentation;

        // Gaussian blur
        cv::GaussianBlur(inputImage, outputBlur, cv::Size(75, 75), 0, 0);
        std::string blurOutputPath = outputFolder + "/" + outputBaseName + "_blur.jpg";
        cv::imwrite(blurOutputPath, outputBlur);

        // Grayscale conversion
        cv::cvtColor(inputImage, outputGrayscale, cv::COLOR_BGR2GRAY);
        std::string grayscaleOutputPath = outputFolder + "/" + outputBaseName + "_grayscale.jpg";
        cv::imwrite(grayscaleOutputPath, outputGrayscale);

        // Edge detection using Canny
        cv::Canny(outputGrayscale, outputEdges, 100, 200);
        std::string edgesOutputPath = outputFolder + "/" + outputBaseName + "_edges.jpg";
        cv::imwrite(edgesOutputPath, outputEdges);

        // Ensure the input image is in 8-bit 3-channel format (CV_8UC3) for watershed transformation
        cv::Mat inputForWatershed;
        if (inputImage.type() == CV_8UC3) {
            inputForWatershed = inputImage.clone();
        } else {
            cv::cvtColor(inputImage, inputForWatershed, cv::COLOR_GRAY2BGR);
        }

        // Create marker image for watershed transformation
        cv::Mat markers(inputForWatershed.size(), CV_32S, cv::Scalar(-1));

        // Image segmentation using watershed transformation
        cv::watershed(inputForWatershed, markers);

        // Convert the marker image to 8-bit single-channel format for visualization
        markers.convertTo(outputSegmentation, CV_8U);

        std::string segmentationOutputPath = outputFolder + "/" + outputBaseName + "_segmentation.jpg";
        cv::imwrite(segmentationOutputPath, outputSegmentation);

        cv::Mat outputMorphology;

        // Create a large kernel for morphological transformation
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(75, 75));

        // Apply morphological transformation (dilation) using the large kernel
        cv::morphologyEx(inputImage, outputMorphology, cv::MORPH_DILATE, kernel);
        std::string morphologyOutputPath = outputFolder + "/" + outputBaseName + "_morphology.jpg";
        cv::imwrite(morphologyOutputPath, outputMorphology);

        cv::Mat outputHeavyBlur;

        // Apply a heavy Gaussian blur with a large kernel size (e.g., 25x25)
        cv::GaussianBlur(inputImage, outputHeavyBlur, cv::Size(75, 75), 30, 30);

        std::string heavyBlurOutputPath = outputFolder + "/" + outputBaseName + "_heavy_blur.jpg";
        cv::imwrite(heavyBlurOutputPath, outputHeavyBlur);
    }

    // Get the current progress
    int getProgress() const {
        return progress.load();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_folder> <output_folder>" << std::endl;
        return 1;
    }

    std::string folderPath = argv[1]; // Input folder path
    std::string outputFolder = argv[2]; // Output folder path

    // Create the output folder if it doesn't exist
    fs::create_directory(outputFolder);

    // Initialize progress bar
    int totalImages = std::distance(fs::directory_iterator(folderPath), fs::directory_iterator());
    std::atomic<int> progress(0);

    // Start measuring overall processing time
    auto start_total_time = high_resolution_clock::now();

    // Perform image processing on images in the input folder using TBB for parallelism
    ImageProcessingTask task(outputFolder);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, totalImages),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                if (progress.load() >= totalImages) {
                    return;
                }
                fs::directory_entry entry = *std::next(fs::directory_iterator(folderPath), i);
                task(entry);
                progress.fetch_add(1);
                int current = progress.load();
                int total = totalImages;
                float percent = static_cast<float>(current) / static_cast<float>(total) * 100.0;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << percent << "%";
                std::cout.flush();
            }
        }
    );

    // Stop measuring overall processing time
    auto end_total_time = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(end_total_time - start_total_time);

    std::cout << "\nTotal processing time: " << total_duration.count() / 60000.0 << " minutes" << std::endl;

    return 0;
}
