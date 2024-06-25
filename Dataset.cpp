#include "Dataset.h"

CIFAR10Dataset::CIFAR10Dataset(const std::string &root, bool train)
    : root_(root), train_(train)
{
    this->PrepareData();
    if (train_)
    {
        this->AugmentData();
    }
}

torch::data::Example<> CIFAR10Dataset::get(size_t index)
{
    return {images_[index], targets_[index]};
}

torch::optional<size_t> CIFAR10Dataset::size() const
{
    return images_.size(0);
}

void CIFAR10Dataset::PrepareData()
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

    std::cout << "Data Loaded" << std::endl;
}

void CIFAR10Dataset::AugmentData()
{
    if (!train_)
    {
        // Don't augment test data
        return;
    }

    std::vector<torch::Tensor> augmented_images;
    std::vector<torch::Tensor> augmented_targets;

    for (int64_t i = 0; i < images_.size(0); ++i)
    {
        torch::Tensor image = images_[i];
        torch::Tensor target = targets_[i];

        // Original image
        augmented_images.push_back(image);
        augmented_targets.push_back(target);

        // Horizontal flip (50% chance)
        if (rand() % 2 == 0)
        {
            augmented_images.push_back(image.flip(2));
            augmented_targets.push_back(target);
        }

        // Random crop (padding then crop)
        {
            int padding = 4;
            torch::Tensor padded = torch::constant_pad_nd(image, {padding, padding, padding, padding}, 0);
            int h_offset = rand() % (2 * padding);
            int w_offset = rand() % (2 * padding);
            torch::Tensor cropped = padded.slice(1, h_offset, h_offset + 32).slice(2, w_offset, w_offset + 32);
            augmented_images.push_back(cropped);
            augmented_targets.push_back(target);
        }

        // Random rotation (-15 to 15 degrees)
        {
            float angle = (rand() % 31 - 15) * M_PI / 180.0;
            cv::Mat cv_image(32, 32, CV_32FC3, image.permute({1, 2, 0}).contiguous().data_ptr<float>());
            cv::Mat rotated;
            cv::Point2f center(16.0, 16.0);
            cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle * 180.0 / M_PI, 1.0);
            cv::warpAffine(cv_image, rotated, rotation_matrix, cv_image.size());
            torch::Tensor rotated_tensor = torch::from_blob(rotated.data, {32, 32, 3}, torch::kFloat32).permute({2, 0, 1});
            augmented_images.push_back(rotated_tensor);
            augmented_targets.push_back(target);
        }
    }

    images_ = torch::stack(augmented_images);
    targets_ = torch::stack(augmented_targets);

    std::cout << "Data Prepared" << std::endl;
}