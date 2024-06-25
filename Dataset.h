#pragma once
#include <fstream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class CIFAR10Dataset : public torch::data::datasets::Dataset<CIFAR10Dataset>
{
public:
    explicit CIFAR10Dataset(const std::string &root, bool train = true);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    void PrepareData();
    void AugmentData();

private:
    std::string root_;
    bool train_;
    torch::Tensor images_;
    torch::Tensor targets_;
};