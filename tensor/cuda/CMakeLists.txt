# packages
find_package(CUDA)

#include_directories ("${PROJECT_SOURCE_DIR}")

# nvcc flags
set(CUDA_NVCC_FLAGS -O3)

if (CMAKE_BUILD_TYPE MATCHES Debug)
	list(APPEND CUDA_NVCC_FLAGS "-G")
	list(APPEND CUDA_NVCC_FLAGS "-g")
endif()

list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
list(APPEND CUDA_NVCC_FLAGS "-arch=compute_61;-code=sm_61;")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60")

#list(APPEND CUDA_NVCC_FLAGS "code=sm_52")
#set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)


file(GLOB_RECURSE CURRENT_HEADERS  *.cuh)
file(GLOB CURRENT_SOURCES  *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(cuda SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_library(cuda STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_cublas_to_target(cuda)
