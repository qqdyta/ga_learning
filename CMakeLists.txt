cmake_minimum_required(VERSION 3.29)
project(cuda CUDA)

# 查找OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda main.cu)

# 将OpenCV的包含目录加入到目标属性中
target_include_directories(cuda PRIVATE ${OpenCV_INCLUDE_DIRS})

# 链接OpenCV库
target_link_libraries(cuda PRIVATE ${OpenCV_LIBS})

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)