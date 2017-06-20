cmake_minimum_required(VERSION 2.8.8)

# ---[ Solution name
project(kfusion C CXX)

# ---[ Configurations types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
if (DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

# ---[ utility
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/find_cuda.cmake)

# ---[ find dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA 8.0 REQUIRED)
find_package(OpenNI REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OPENNI_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  # ${EIGEN3_INCLUDE_DIR}           # #include <eigen3/Eigen/Core>
  "/usr/include/eigen3/"        # #include <Eigen/Core>
)

# ---[ misc settings
if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# ---[ cuda settings
set(HAVE_CUDA 1)
#list(APPEND CUDA_NVCC_FLAGS "-arch;compute_20")
list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_61,code=sm_61")

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--disable-warnings;--ptxas-options=-v;-use_fast_math;-lineinfo;-std=c++11)

# warnings_disable(CMAKE_CXX_FLAGS /wd4985)

add_subdirectory(kfusion)
add_subdirectory(apps)
add_subdirectory(test)
