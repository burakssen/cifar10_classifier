#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <fstream>

class CIFAR10Dataset : public torch::data::datasets::Dataset<CIFAR10Dataset>
{
public:
    explicit CIFAR10Dataset(const std::string &root, bool train = true)
        : root_(root), train_(train)
    {
        this->PrepareData();
    }

    torch::data::Example<> get(size_t index) override
    {
        return {images_[index], targets_[index]};
    }

    torch::optional<size_t> size() const override
    {
        return images_.size(0);
    }

private:
    void PrepareData()
    {
        std::vector<std::string> files;
        if (train_)
            files = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"};
        else
            files = {"test_batch.bin"};

        std::vector<torch::Tensor> images;
        std::vector<torch::Tensor> targets;

        for (const auto &file : files)
        {
            std::ifstream file_stream{
                root_ + "/" + file,
                std::ios::binary};

            if (!file_stream)
            {
                throw std::runtime_error("Cannot open file: " + root_ + "/" + file);
            }

            std::vector<uint8_t> data(
                (std::istreambuf_iterator<char>(file_stream)),
                std::istreambuf_iterator<char>());

            const size_t num_images = data.size() / (32 * 32 * 3 + 1);
            for (size_t i = 0; i < num_images; ++i)
            {
                const size_t offset = i * (32 * 32 * 3 + 1);
                const uint8_t target = data[offset];
                auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
                auto tensor = torch::from_blob(data.data() + offset + 1, {3, 32, 32}, options);
                tensor = tensor.to(torch::kFloat32) / 255;
                images.push_back(tensor);
                targets.push_back(torch::full({1}, target, options.dtype(torch::kLong)));
            }
        }

        images_ = torch::stack(images);
        targets_ = torch::stack(targets);
    }

private:
    std::string root_;
    bool train_;
    torch::Tensor images_;
    torch::Tensor targets_;
};

class CIFAR10Classifier : public torch::nn::Module
{
public:
    CIFAR10Classifier()
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

    torch::Tensor forward(torch::Tensor x)
    {
        x = features->forward(x);
        x = x.view({x.size(0), -1});
        x = classifier->forward(x);
        return x;
    }

private:
    torch::nn::Sequential features{nullptr}, classifier{nullptr};
};

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

        auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-4));

        auto criterion = torch::nn::CrossEntropyLoss();

        // Training
        for (size_t epoch = 1; epoch <= 50; ++epoch)
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
