#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "Dataset.h"
#include "Classifier.h"

int main(int argc, char **argv)
{
    if (!torch::mps::is_available())
    {
        std::cerr << "MPS is not available. Please check your environment." << std::endl;
        return 1;
    }

    if (argc != 2)
    {
        std::cerr << "Usage: cifar10_classifier <command>" << std::endl;
        std::cerr << "Available commands: train, test" << std::endl;
        return 1;
    }

    std::string command = argv[1];

    torch::Device device = torch::kMPS;

    auto model = std::make_shared<CIFAR10Classifier>();

    if (command == "train")
    {
        auto train_dataset = CIFAR10Dataset("../resources/train", true).map(torch::data::transforms::Stack<>());
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(64).workers(8));

        model->to(device);

        auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-3));

        auto criterion = torch::nn::CrossEntropyLoss();

        // Training
        for (size_t epoch = 1; epoch <= 20; ++epoch)
        {
            model->train();
            size_t batch_index = 0;

            for (auto &batch : *train_loader)
            {
                auto data = batch.data.to(device);
                auto targets = batch.target.to(device).view({-1});

                optimizer.zero_grad();
                auto output = model->forward(data);
                auto loss = criterion(output, targets);
                loss.backward();
                optimizer.step();

                if (++batch_index % 100 == 0)
                {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << std::endl;
                }
            }
        }

        torch::save(model, "model.pt");

        auto test_dataset = CIFAR10Dataset("../resources/test", false).map(torch::data::transforms::Stack<>());
        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(64).workers(8));

        // Testing
        model->eval();
        torch::NoGradGuard no_grad;
        size_t correct = 0;
        size_t total = 0;

        for (const auto &batch : *test_loader)
        {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device).view({-1});

            auto output = model->forward(data);
            auto predictions = output.argmax(1);
            correct += predictions.eq(targets).sum().item<int>();
            total += data.size(0);
        }

        std::cout << "Accuracy: " << (100.0 * correct / total) << "%" << std::endl;
    }
    else
    {
        torch::load(model, "model.pt");

        auto test_dataset = CIFAR10Dataset("../resources/test", false).map(torch::data::transforms::Stack<>());
        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(64).workers(8));

        std::vector<std::string> class_names = {
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"};

        // Testing
        model->eval();
        torch::NoGradGuard no_grad;
        size_t correct = 0;
        size_t total = 0;

        for (const auto &batch : *test_loader)
        {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device).view({-1});

            auto output = model->forward(data);
            auto predictions = output.argmax(1);
            correct += predictions.eq(targets).sum().item<int>();
            total += data.size(0);

            if (total % 100 == 0)
            {
                for (int i = 0; i < data.size(0); i++)
                {
                    auto image = data[i].to(torch::kCPU).mul(255).clamp(0, 255).to(torch::kU8).permute({1, 2, 0}).contiguous();
                    cv::Mat mat(cv::Size(32, 32), CV_8UC3, image.data_ptr());
                    int pred = predictions[i].item<int>();
                    std::string label = "Pred: " + class_names[pred] + " | True: " + class_names[targets[i].item<int>()];
                    std::cout << label << std::endl;
                    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
                    cv::imshow("Image", mat);
                    cv::waitKey(0);
                }
            }
        }

        std::cout << "Accuracy: " << (100.0 * correct / total) << "%" << std::endl;
    }

    return 0;
}
