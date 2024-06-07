cmake_minimum_required(VERSION 3.28)
project(cifar10_classifier)

set(CMAKE_CXX_STANDARD 23)

find_package(Torch REQUIRED)

# Set OpenCV installation directory
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/opencv/4.9.0_9.reinstall")

# Find OpenCV package
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs REQUIRED)

add_executable(cifar10_classifier main.cpp)

# Include OpenCV headers
target_include_directories(cifar10_classifier PUBLIC ${OpenCV_INCLUDE_DIRS})


target_link_libraries(cifar10_classifier
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
)