#pragma once

#include <fstream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class CIFAR10Classifier : public torch::nn::Module
{
public:
    CIFAR10Classifier();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential features{nullptr}, classifier{nullptr};
};