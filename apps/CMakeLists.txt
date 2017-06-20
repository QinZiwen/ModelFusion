include_directories(${CMAKE_SOURCE_DIR}/kfusion/include)

add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS} kfusion)

set_target_properties(demo PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(offLine offLine.cpp)
target_link_libraries(offLine ${OpenCV_LIBS} kfusion)

set_target_properties(offLine PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    
add_executable(orbslam2TSDF orbslam2TSDF.cpp)
target_link_libraries(orbslam2TSDF ${OpenCV_LIBS} kfusion libORB_SLAM2.so ${Pangolin_LIBRARIES})

set_target_properties(orbslam2TSDF PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")