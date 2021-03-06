set(EXECUTABLE ethash-cu)

FIND_PACKAGE(CUDA REQUIRED)

file(GLOB SRC_LIST "*.cpp" "*.cu")
file(GLOB HEADERS "*.h" "*.cuh")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++11;--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo)

LIST(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)
LIST(APPEND CUDA_NVCC_FLAGS_DEBUG -G)

if(COMPUTE)
	LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_${COMPUTE},code=sm_${COMPUTE})
else(COMPUTE)
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20;-gencode arch=compute_30,code=sm_30;-gencode arch=compute_32,code=sm_32;-gencode arch=compute_35,code=sm_35;-gencode arch=compute_50,code=sm_50;-gencode arch=compute_52,code=sm_52)
endif(COMPUTE)



include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)
CUDA_ADD_LIBRARY(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${CUDA_LIBRARIES} ethash)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

