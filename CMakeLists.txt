set(PROJECT "FloydWarshall")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# ------------------------------------------------------------------------------
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP)

if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
# ------------------------------------------------------------------------------

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if (COMPILER_SUPPORTS_CXX14)
    add_compile_options("-std=c++14")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++14 support.")
endif()

# ------------------------------------------------------------------------------

#add_compile_options("-O3")
# add_compile_options("-pg")

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp
											 ${PROJECT_SOURCE_DIR}/src/*.cu
)

# ------------------------------------------------------------------------------
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-w")  #"--resource-usage"
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_61,code=sm_61)  # change architecture here for your GPU
set(CUDA_SEPARABLE_COMPILATION ON)
cuda_add_executable(floydwarshall ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
# ------------------------------------------------------------------------------


# add_executable(floydwarshall ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
