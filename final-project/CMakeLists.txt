cmake_minimum_required(VERSION 3.10)
project(arg_min_max_parallel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找所有源文件
file(GLOB SRC_FILES
    "*.cpp"
    "*.c"
    "*.cu"
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

# OpenMP 支持
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /openmp)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp -mavx2 -mfma)
    target_link_libraries(${PROJECT_NAME} PRIVATE -fopenmp)
endif()

# AVX2 支持
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -mavx2 -mfma)
endif()

# pthread 支持（Linux/macOS）
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PUBLIC pthread)
endif()

# 头文件目录（如有自定义头文件/第三方库）
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 可选：编译警告
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O0)
endif()