#include "Classifier.h"

CIFAR10Classifier::CIFAR10Classifier()
{
    // Define the feature extractor
    features = torch::nn::Sequential(
        // Block 1
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        // Block 2
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        // Block 3
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),

        // Block 4
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)),
        torch::nn::BatchNorm2d(512),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
        torch::nn::BatchNorm2d(512),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
        torch::nn::BatchNorm2d(512),
        torch::nn::ReLU(true),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

    // Define the classifier
    classifier = torch::nn::Sequential(
        torch::nn::Dropout(0.5),
        torch::nn::Linear(512 * 2 * 2, 4096),
        torch::nn::ReLU(true),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(4096, 4096),
        torch::nn::ReLU(true),
        torch::nn::Linear(4096, 10));

        // Register the modules
    register_module("features", features);
    register_module("classifier", classifier);
}

torch::Tensor CIFAR10Classifier::forward(torch::Tensor x)
{
    x = features->forward(x);
    x = x.view({x.size(0), -1});
    x = classifier->forward(x);
    return x;
}